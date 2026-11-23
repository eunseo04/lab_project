"""Observable, per-Blue case history; never reads simulator or adversary state."""
from collections import Counter
from copy import deepcopy
import json

USER_IOCS = {"cmd.sh", "cmd.exe"}
ADMIN_IOCS = {"escalate.sh", "escalate.exe"}
HOST_ACTIONS = {"Analyse", "Remove", "Restore", "DeployDecoy"}


def visible_evidence(body):
    """Use the same public PID/file indicators as the original ACD formatter."""
    pids = {str(p["PID"]) for p in body.get("Processes", [])
            if isinstance(p, dict) and "PID" in p and "username" not in p}
    files = {f.get("File Name") for f in body.get("Files", []) if isinstance(f, dict)}
    iocs = files & (USER_IOCS | ADMIN_IOCS)
    level = 3 if iocs & ADMIN_IOCS else 2 if iocs & USER_IOCS else 1 if pids else 0
    return level, pids, iocs


def chosen_action(row):
    text = row.get("requested_action") or "Sleep"
    if row.get("response") and not row.get("parse_error"):
        try:
            text = json.loads(row["response"])["action"]
        except (ValueError, KeyError, TypeError):
            pass
    parts = text.split()
    name = parts[0] if parts else "Sleep"
    canonical={value.lower():value for value in HOST_ACTIONS | {"Sleep","BlockTrafficZone","AllowTrafficZone"}}
    name=canonical.get(name.lower(),name)
    target = parts[-1].removeprefix("host:").removeprefix("subnet:") if len(parts)>1 else None
    return name, target


def readable_context(context):
    """Render evidence as facts, never substitute a scripted primitive action."""
    pending=context.get("pending_action")
    lines=["# OBSERVED CASE HISTORY FROM THIS EPISODE",
        "These are retained observations, not hidden ground truth. A current tick with no new alerts does not cancel an unreviewed earlier case.",
        "Availability now: an own action is still IN_PROGRESS." if pending else
        "Availability now: FREE. There is no own action IN_PROGRESS; earlier completed actions are not still running."]
    for case in context.get("open_cases",[]):
        severity={0:"an earlier own response awaiting review",1:"earlier WARNING, a flagged local process",2:"earlier ALERT, a user-level file indicator",3:"earlier CRITICAL, an admin-level file indicator"}[case["observed_level"]]
        facts=[f"Host {case['host']}: {severity}; observed {case['evidence_age']} steps ago",
               "flagged PIDs="+','.join(case["flagged_pids"]),"file indicators="+','.join(case["file_indicators"])]
        for key in ("last_analysis","last_remove","last_restore"):
            operation=case.get(key)
            if operation:facts.append(f"{operation['name']} completed at step {operation['completed_at']} with status {operation['status']}")
        if case["verification_due"]:
            facts.append("follow-up diagnosis after Remove has NOT been reviewed; the completed Remove call alone does not establish cleanup")
        elif case["observed_level"]>=3:
            facts.append("no later successful Restore has closed this admin-indicator case")
        elif case["response_due"]:
            facts.append("no response has yet been completed for this evidence version")
        lines.append("- "+"; ".join(facts)+".")
    if not context.get("open_cases"):lines.append("No unreviewed local cases are retained. Quiet Sleep remains available.")
    if "investigation_leads" in context:
        lines.append("Connection leads below are observed network activity, not proof of compromise. They can justify bounded information gathering; no PID alert does not certify a safe host.")
        for lead in context["investigation_leads"]:
            lines.append(f"- Connection lead on {lead['host']}: last seen {lead['age']} steps ago; observed endpoints={json.dumps(lead['endpoints'])}; last successful review={lead['reviewed_at']}; investigation_due={lead['investigation_due']}; cooldown_remaining={lead['cooldown_remaining']}.")
    if context.get("additional_open_cases"):lines.append(f"Additional retained cases: {context['additional_open_cases']}; highest-severity cases are listed above.")
    lines.append("Successful own decoys: "+json.dumps(context.get("successful_decoys",{})))
    lines.append("Successful own blocks still recorded: "+json.dumps(context.get("own_observed_blocks",[])))
    lines.append("Decoded peer messages (busy alone is not a threat): "+json.dumps(context.get("peers",[])))
    lines.append("Old cmd files may remain after Remove. A reviewed case is a workflow status, not proof the simulator host is clean.")
    return "\n".join(lines)


class ObservableCaseLedger:
    """Keeps transient alerts and asynchronous results from this agent's trace.

    A completed Remove call does not certify that malware was removed. File IOCs
    can remain after Remove, so an unchanged cmd.sh alone is not new compromise.
    Closing a reviewed case is a workflow fact, never a hidden-state clean label.
    """
    def __init__(self, agent, connection_leads=False):
        self.agent = agent
        self.allowed_hosts = set()
        self.cases = {}
        self.pending = None
        self.completed = []
        self.decoys = Counter()
        self.blocks = set()
        self.last_context = None
        self.connection_leads = connection_leads
        self.leads = {}

    def _observe_connections(self, host, body, step):
        # CAGE4 observations can omit either address or either port. The event
        # itself remains visible evidence of activity, not proof of compromise.
        endpoints={}
        for process in body.get("Processes",[]) or []:
            if not isinstance(process,dict):continue
            for connection in process.get("Connections",[]) or []:
                if not isinstance(connection,dict):continue
                fields={str(k):deepcopy(v) for k,v in connection.items() if v is not None and v!=""}
                if not fields:continue
                for field in ("remote_address","local_address","remote_port","local_port"):
                    if field in fields:fields[field]=str(fields[field])
                remote=fields.get("remote_address","");local=fields.get("local_port","")
                # Preserve the complete-endpoint v3 representation and its
                # remote ephemeral-port deduplication. Partial events retain
                # the actual supplied fields; missing values are not invented.
                if remote and local:
                    visible={"remote_address":remote,"local_port":local}
                else:visible=fields
                key=(remote,local,fields.get("remote_port","") if not local else "",
                    fields.get("local_address","") if not remote else "")
                if not any(key):key=("","","",json.dumps(fields,sort_keys=True))
                endpoints[key]=visible
        if not endpoints:
            return
        lead = self.leads.setdefault(host, {"host":host, "last_seen":step,
            "reviewed_at":None, "endpoints":{}})
        lead["last_seen"] = step
        for endpoint,fields in endpoints.items():
            lead["endpoints"][endpoint] = {"last_seen":step,"fields":fields}

    def _case(self, host):
        return self.cases.setdefault(host, {"host":host, "level":0, "version":0,
            "pids":set(), "iocs":set(), "active":False, "last_evidence":-1,
            "last_new_pid":-1,
            "responded_version":-1, "analysed_version":-1, "verify_due":False,
            "last_analysis":None, "last_remove":None, "last_restore":None})

    def observe(self, raw, step):
        for host, body in raw.items():
            if host not in self.allowed_hosts or not isinstance(body, dict):
                continue
            if self.connection_leads:
                self._observe_connections(host, body, step)
            level, pids, iocs = visible_evidence(body)
            if not level:
                continue
            case = self._case(host)
            fresh = bool(pids-case["pids"] or iocs-case["iocs"])
            if fresh:
                case["version"] += 1
                case["last_evidence"] = step
                case["active"] = True
            if pids-case["pids"]:case["last_new_pid"]=step
            case["level"] = max(case["level"], level)
            case["pids"].update(pids)
            case["iocs"].update(iocs)

    def context(self, state, step):
        self.allowed_hosts.update(state["hosts"])
        self.observe(state["observation"], step)
        cases=[]
        for case in self.cases.values():
            if not case["active"] and not case["verify_due"]:
                continue
            cases.append({"host":case["host"], "observed_level":case["level"],
                "evidence_version":case["version"], "evidence_age":step-case["last_evidence"],
                "flagged_pids":sorted(case["pids"])[-4:], "file_indicators":sorted(case["iocs"]),
                "response_due":case["version"]>case["responded_version"],
                "verification_due":case["verify_due"],
                "analysis_due":case["version"]>case["analysed_version"],
                "last_analysis":case["last_analysis"], "last_remove":case["last_remove"],
                "last_restore":case["last_restore"]})
        cases.sort(key=lambda c:(-c["observed_level"],not c["verification_due"],-c["evidence_age"],c["host"]))
        peers=[]
        ids=[i for i in range(5) if i!=int(self.agent.rsplit("_",1)[-1])]
        for idx, bits in zip(ids,state["observation"].get("message",[])):
            if len(bits)==8:
                peers.append({"agent":f"blue_agent_{idx}","detected_origin_agents":[j for j in range(5) if bits[j]],
                    "reported_compromise":int(bits[5])*2+int(bits[6]),"busy":bool(bits[7])})
        self.last_context={"source":"this Blue's observed events and own action outcomes only",
            "step":step,"open_cases":cases[:12],"additional_open_cases":max(0,len(cases)-12),
            "pending_action":deepcopy(self.pending),"recent_completed_actions":deepcopy(self.completed[-4:]),
            "successful_decoys":dict(self.decoys),"own_observed_blocks":sorted(self.blocks),"peers":peers,
            "interpretation":"Level 1 is a flagged PID; 2/3 are observed user/admin file indicators. "
                "TRUE means an action completed, not proof a threat disappeared. Old file artifacts may persist. "
                "An absent current alert does not erase an unreviewed earlier observation. "
                "Busy Sleep is automatic; a free agent retains the same primitive actions as the baseline."}
        if self.connection_leads:
            leads=[]
            for host, lead in self.leads.items():
                age=step-lead["last_seen"]
                if age>32:
                    continue
                reviewed=lead["reviewed_at"]
                cooldown=max(0, 16-(step-reviewed)) if reviewed is not None else 0
                due=(reviewed is None or lead["last_seen"]>reviewed) and cooldown==0
                endpoints=[deepcopy(record["fields"])
                    for ep,record in sorted(lead["endpoints"].items()) if step-record["last_seen"]<=32]
                leads.append({"host":host, "age":age, "reviewed_at":reviewed,
                    "endpoints":endpoints[:3], "investigation_due":due,
                    "cooldown_remaining":cooldown})
            leads.sort(key=lambda x:(not x["investigation_due"], x["age"], x["host"]))
            self.last_context["investigation_leads"]=leads[:8]
            self.last_context["additional_investigation_leads"]=max(0,len(leads)-8)
            self.last_context["connection_schema"]="optional_endpoint_fields_v2"
        return deepcopy(self.last_context)

    def transition(self, row):
        step=row["step"]
        before=deepcopy(self.last_context or {})
        name,target=chosen_action(row)
        applied=bool(row.get("policy_invoked") and row.get("request_applied") and not row.get("parse_error"))
        case=self.cases.get(target)
        version=case["version"] if case else 0
        repeats=[a for a in self.completed if a["name"]==name and a.get("target")==target
                 and a.get("evidence_version")==version]
        diagnostics={"free_policy_decision":bool(row.get("policy_invoked")),
            "automatic_busy_wait":not row.get("policy_invoked") and name=="Sleep",
            "sleep_with_open_case":applied and name=="Sleep" and bool(before.get("open_cases")),
            "repeated_analysis_same_evidence":applied and name=="Analyse" and bool(repeats),
            "analysis_before_unhandled_flagged_pid":applied and name=="Analyse" and bool(case and
                case["pids"] and case["level"]<3 and case["version"]>case["responded_version"]),
            "duplicate_successful_decoy":applied and name=="DeployDecoy" and self.decoys[target]>0}
        if applied and name in HOST_ACTIONS | {"BlockTrafficZone","AllowTrafficZone"}:
            if self.pending is not None:
                raise RuntimeError("An available policy decision must not overwrite an unfinished own action")
            self.pending={"name":name,"target":target,"started_at":step,"evidence_version":version}
        self.observe(row["observation_after"],step+1)
        status=str(row["observation_after"].get("success","UNKNOWN"))
        completed=None
        if self.pending is not None and status in ("TRUE","FALSE"):
            completed={**self.pending,"completed_at":step+1,"status":status}
            self.completed.append(completed)
            name,target=completed["name"],completed["target"]
            if (self.connection_leads and status=="TRUE" and name in {"Analyse","Restore"}
                    and target in self.leads):
                self.leads[target]["reviewed_at"]=step+1
            case=self.cases.get(target)
            if name in {"Analyse","Remove","Restore"} and target in self.allowed_hosts:
                case=self._case(target)
                if name=="Remove":
                    case["last_remove"]=deepcopy(completed)
                    if status=="TRUE":
                        case["responded_version"]=max(case["responded_version"],completed["evidence_version"])
                    case["verify_due"]=True
                    case["active"]=True
                elif name=="Restore":
                    case["last_restore"]=deepcopy(completed)
                    if status=="TRUE" and case["version"]<=completed["evidence_version"]:
                        case["active"]=case["verify_due"]=False
                        case["level"]=0
                else:
                    body=row["observation_after"].get(target,{})
                    level,_,iocs=visible_evidence(body)
                    completed["reported_file_indicators"]=sorted(iocs)
                    case["last_analysis"]=deepcopy(completed)
                    new_pid=case["last_new_pid"]>completed["started_at"]
                    analysed=completed["evidence_version"] if new_pid else case["version"]
                    case["analysed_version"]=max(case["analysed_version"],analysed)
                    if status=="TRUE":
                        newer=new_pid
                        removed=case["last_remove"]
                        after_removal=bool(removed and removed["status"]=="TRUE"
                            and case["last_new_pid"]<=removed["started_at"])
                        if level>=3:
                            case["active"]=True
                        elif not newer and (after_removal or level==0):
                            case["active"]=False
                        case["verify_due"]=False
            if status=="TRUE" and name=="DeployDecoy":self.decoys[target]+=1
            if status=="TRUE" and name=="BlockTrafficZone":self.blocks.add(target)
            if status=="TRUE" and name=="AllowTrafficZone":self.blocks.discard(target)
            self.pending=None
        row["behavior_context"]=before
        row["behavior_outcome"]={**diagnostics,"completed_action":deepcopy(completed)}
        return row["behavior_outcome"]
