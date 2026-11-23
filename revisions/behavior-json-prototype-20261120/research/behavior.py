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
    target = parts[-1].removeprefix("host:").removeprefix("subnet:") if len(parts)>1 else None
    return name, target


class ObservableCaseLedger:
    """Keeps transient alerts and asynchronous results from this agent's trace.

    A completed Remove call does not certify that malware was removed. File IOCs
    can remain after Remove, so an unchanged cmd.sh alone is not new compromise.
    Closing a reviewed case is a workflow fact, never a hidden-state clean label.
    """
    def __init__(self, agent):
        self.agent = agent
        self.allowed_hosts = set()
        self.cases = {}
        self.pending = None
        self.completed = []
        self.decoys = Counter()
        self.blocks = set()
        self.last_context = None

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
            case=self.cases.get(target)
            if name in {"Analyse","Remove","Restore"} and target in self.allowed_hosts:
                case=self._case(target)
                if name=="Remove":
                    case["last_remove"]=deepcopy(completed)
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
                        after_removal=case["last_remove"] is not None
                        if level>=3:
                            case["active"]=True
                        elif not newer and (after_removal or level<2):
                            case["active"]=False
                        case["verify_due"]=False
            if status=="TRUE" and name=="DeployDecoy":self.decoys[target]+=1
            if status=="TRUE" and name=="BlockTrafficZone":self.blocks.add(target)
            if status=="TRUE" and name=="AllowTrafficZone":self.blocks.discard(target)
            self.pending=None
        row["behavior_context"]=before
        row["behavior_outcome"]={**diagnostics,"completed_action":deepcopy(completed)}
        return row["behavior_outcome"]
