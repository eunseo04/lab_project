import docker
import ipaddress

#=========================
# Docker Network 설정
#=========================

class NetworkManager:
    def __init__(self, network_name, subnet):
        self.network_name = network_name
        self.subnet = subnet
        # Docker Engine에 연결하는 Docker 클라이언트 생성
        self.client = docker.from_env() 
        
    # 네트워크 생성    
    def create_network(self):
        # IPAM(IP Address Management) 설정을 통해 서브넷 지정
        ipam_config = docker.types.IPAMConfig(
            pool_configs=[
                docker.types.IPAMPool(subnet=self.subnet)
            ]
        )
        network = self.client.networks.create(
            self.network_name,
            driver='bridge', # Docker network에 붙어 있는 컨테이너들이 서로 통신
            ipam=ipam_config
        )
        return network
    
    # 네트워크 존재 여부 확인
    def get_network(self):
        try:
            return self.client.networks.get(self.network_name)
        except docker.errors.NotFound:
            return None

    # 네트워크 제거
    def remove_network(self):
        try:
            network = self.client.networks.get(self.network_name)
            network.remove()
            print(f"Network '{self.network_name}' removed successfully.")
        except docker.errors.NotFound:
            print(f"Network '{self.network_name}' not found.")
        except docker.errors.APIError as e:
            print(f"Failed to remove network '{self.network_name}': {e}")

    # subnet에서 사용할 IP 하나 선택
    def get_available_ip(self):
        """
        Get an available IP address from the subnet.
        """
        network = ipaddress.IPv4Network(self.subnet)
        for ip in network.hosts():
            # Skip gateway IPs (usually .1, .254)
            if ip.packed[-1] in [1, 254]:
                continue
            return str(ip)
        return None