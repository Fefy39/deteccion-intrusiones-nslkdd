"""Configuración central del proyecto (rutas, columnas y mapeo de clases)."""
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
DATOS_RAW = RAIZ / "data" / "raw"
DATOS_PROC = RAIZ / "data" / "processed"
MODELOS = RAIZ / "models"

URL_BASE = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/"
ARCHIVOS = {"train": "KDDTrain+.txt", "test": "KDDTest+.txt"}
SEMILLA = 42

COLUMNAS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land",
    "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in", "num_compromised",
    "root_shell", "su_attempted", "num_root", "num_file_creations", "num_shells",
    "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login", "count",
    "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
    "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
    "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
    "label", "difficulty",
]
CATEGORICAS = ["protocol_type", "service", "flag"]

# Agrupación de los ataques en 4 categorías (Tavallaee et al., 2009)
CATEGORIA_ATAQUE = {
    "normal": "normal",
    # DoS
    "back": "dos", "land": "dos", "neptune": "dos", "pod": "dos", "smurf": "dos",
    "teardrop": "dos", "apache2": "dos", "mailbomb": "dos", "processtable": "dos",
    "udpstorm": "dos",
    # Probe
    "ipsweep": "probe", "nmap": "probe", "portsweep": "probe", "satan": "probe",
    "mscan": "probe", "saint": "probe",
    # R2L
    "ftp_write": "r2l", "guess_passwd": "r2l", "imap": "r2l", "multihop": "r2l",
    "phf": "r2l", "spy": "r2l", "warezclient": "r2l", "warezmaster": "r2l",
    "named": "r2l", "sendmail": "r2l", "snmpgetattack": "r2l", "snmpguess": "r2l",
    "worm": "r2l", "xlock": "r2l", "xsnoop": "r2l", "httptunnel": "r2l",
    # U2R
    "buffer_overflow": "u2r", "loadmodule": "u2r", "perl": "u2r", "rootkit": "u2r",
    "ps": "u2r", "sqlattack": "u2r", "xterm": "u2r",
}
CLASES = ["normal", "dos", "probe", "r2l", "u2r"]
