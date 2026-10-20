#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_FILE="${LOG_FILE:-${HOME}/rocky9-minikube-setup-$(date '+%Y%m%d-%H%M%S').log}"
mkdir -p "$(dirname "$LOG_FILE")"
exec > >(tee -a "$LOG_FILE") 2>&1

log() {
    printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

fail() {
    log "ERROR: $*"
    exit 1
}

on_error() {
    local status=$?
    log "ERROR: command failed near line ${BASH_LINENO[0]} (exit ${status}); see ${LOG_FILE}"
    exit "$status"
}
trap on_error ERR

if [[ ${EUID} -eq 0 ]]; then
    fail "Run this as your regular login user, not with sudo. The script uses sudo only for system changes."
fi
command -v sudo >/dev/null 2>&1 || fail "sudo is required. Install/configure it, then rerun."
sudo -v

[[ -r /etc/os-release ]] || fail "Cannot identify the operating system."
# shellcheck disable=SC1091
source /etc/os-release
[[ "${ID:-}" == "rocky" && "${VERSION_ID%%.*}" == "9" ]] \
    || fail "This script supports Rocky Linux 9 only (detected ${PRETTY_NAME:-unknown})."
[[ -f "${PROJECT_ROOT}/requirements.txt" && -f "${PROJECT_ROOT}/deployment/kubernetes.yaml" ]] \
    || fail "Run this script from a checkout of the heart-disease-mlops project."

case "$(uname -m)" in
    x86_64)
        DOWNLOAD_ARCH="amd64"
        ;;
    aarch64|arm64)
        DOWNLOAD_ARCH="arm64"
        ;;
    *)
        fail "Unsupported CPU architecture: $(uname -m)"
        ;;
esac

log "Rocky Linux 9 bootstrap and Minikube deployment started."
log "Project directory: ${PROJECT_ROOT}"
log "Log file: ${LOG_FILE}"

ensure_rpm() {
    local package="$1"
    if rpm -q --quiet "$package"; then
        log "SKIP: RPM package already installed: ${package}"
    else
        log "INSTALL: RPM package: ${package}"
        sudo dnf install -y "$package"
    fi
}

ensure_rpm dnf-plugins-core
for package in git curl wget ca-certificates tar gzip conntrack-tools socat \
    iproute iptables gcc gcc-c++ make; do
    ensure_rpm "$package"
done

if command -v python3.11 >/dev/null 2>&1; then
    log "SKIP: Python 3.11 already installed ($(python3.11 --version 2>&1))"
else
    log "Python 3.11 is missing; installing Python, pip, and development headers."
    for package in python3.11 python3.11-pip python3.11-devel; do
        ensure_rpm "$package"
    done
fi
command -v python3.11 >/dev/null 2>&1 || fail "Python 3.11 was not found after installation."

if ! command -v docker >/dev/null 2>&1; then
    log "Docker CLI is missing; configuring the official Docker CE repository."
    if [[ ! -f /etc/yum.repos.d/docker-ce.repo ]]; then
        sudo dnf config-manager --add-repo https://download.docker.com/linux/centos/docker-ce.repo
    else
        log "SKIP: Docker CE repository file already exists."
    fi
    for package in docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin; do
        ensure_rpm "$package"
    done
else
    log "SKIP: Docker CLI already installed ($(docker --version))"
fi

if ! command -v docker >/dev/null 2>&1; then
    fail "Docker CLI is still unavailable after installation."
fi
if ! getent group docker >/dev/null 2>&1; then
    log "Creating the Docker group."
    sudo groupadd docker
fi
if ! id -nG | tr ' ' '\n' | grep -qx docker; then
    log "Adding user $(id -un) to the Docker group."
    sudo usermod -aG docker "$(id -un)"
    log "Log out and back in so the new Docker group membership takes effect, then rerun this script."
    exit 2
fi

log "Ensuring Docker is enabled and running."
sudo systemctl enable --now docker
docker info >/dev/null

install_verified_binary() {
    local name="$1"
    local url="$2"
    local destination="$3"
    local temp_dir expected

    temp_dir="$(mktemp -d)"
    curl --fail --silent --show-error --location "$url" -o "${temp_dir}/${name}"
    curl --fail --silent --show-error --location "${url}.sha256" -o "${temp_dir}/${name}.sha256"
    expected="$(awk '{print $1}' "${temp_dir}/${name}.sha256")"
    [[ "$expected" =~ ^[[:xdigit:]]{64}$ ]] || fail "Invalid SHA-256 checksum for ${name}."
    printf '%s  %s\n' "$expected" "${temp_dir}/${name}" | sha256sum --check --status \
        || fail "Checksum verification failed for ${name}."
    sudo install -o root -g root -m 0755 "${temp_dir}/${name}" "$destination"
    rm -rf -- "$temp_dir"
    log "INSTALLED: ${name} at ${destination} (SHA-256 verified)"
}

if command -v kubectl >/dev/null 2>&1; then
    log "SKIP: kubectl already installed ($(kubectl version --client -o yaml | awk '/gitVersion:/ {print $2; exit}'))"
else
    log "kubectl is missing; installing the latest stable release for ${DOWNLOAD_ARCH}."
    KUBECTL_VERSION="$(curl --fail --silent --show-error --location https://dl.k8s.io/release/stable.txt)"
    install_verified_binary kubectl \
        "https://dl.k8s.io/release/${KUBECTL_VERSION}/bin/linux/${DOWNLOAD_ARCH}/kubectl" \
        /usr/local/bin/kubectl
fi

if command -v minikube >/dev/null 2>&1; then
    log "SKIP: Minikube already installed ($(minikube version | head -n 1))"
else
    log "Minikube is missing; installing the latest stable release for ${DOWNLOAD_ARCH}."
    install_verified_binary minikube \
        "https://storage.googleapis.com/minikube/releases/latest/minikube-linux-${DOWNLOAD_ARCH}" \
        /usr/local/bin/minikube
fi

VENV_DIR="${PROJECT_ROOT}/.venv"
if [[ -x "${VENV_DIR}/bin/python" ]]; then
    if ! "${VENV_DIR}/bin/python" -c 'import sys; raise SystemExit(sys.version_info[:2] != (3, 11))'; then
        fail "Existing .venv is not Python 3.11. Rename or remove it, then rerun this script."
    fi
    log "SKIP: Python 3.11 project virtual environment already exists."
else
    log "Creating project virtual environment with Python 3.11."
    python3.11 -m venv "$VENV_DIR"
fi

PYTHON="${VENV_DIR}/bin/python"
log "Checking/installing Python dependencies from requirements.txt."
"$PYTHON" -m pip install --upgrade pip
"$PYTHON" -m pip install -r "${PROJECT_ROOT}/requirements.txt"

if [[ ! -s "${PROJECT_ROOT}/models/heart_model.joblib" ]]; then
    log "Trained model is missing; preparing data and training the model."
    if [[ ! -s "${PROJECT_ROOT}/data/raw/heart.csv" ]]; then
        "$PYTHON" "${PROJECT_ROOT}/scripts/download_data.py"
    else
        log "SKIP: raw dataset already exists."
    fi
    "$PYTHON" "${PROJECT_ROOT}/scripts/make_eda.py"
    "$PYTHON" -m src.train
else
    log "SKIP: trained model already exists."
fi

log "Running the same lint and test checks used by project CI."
(
    cd "$PROJECT_ROOT"
    "$PYTHON" -m ruff check --select F src scripts tests
    "$PYTHON" -m pytest -q
)

MINIKUBE_CPUS="${MINIKUBE_CPUS:-2}"
MINIKUBE_MEMORY="${MINIKUBE_MEMORY:-4096mb}"
log "Ensuring Minikube is started with Docker driver (${MINIKUBE_CPUS} CPUs, ${MINIKUBE_MEMORY} memory)."
minikube start --driver=docker --cpus="$MINIKUBE_CPUS" --memory="$MINIKUBE_MEMORY"

log "Building the application image in Minikube's image store."
minikube image build -t heart-disease-api:1.0 "$PROJECT_ROOT"

log "Applying Kubernetes resources and waiting for the API deployment."
kubectl apply -f "${PROJECT_ROOT}/deployment/kubernetes.yaml"
kubectl rollout status deployment/heart-api --timeout=300s

MINIKUBE_IP="$(minikube ip)"
API_URL="http://${MINIKUBE_IP}:30080"
log "Checking API health at ${API_URL}/health"
if ! curl --fail --silent --show-error --retry 20 --retry-connrefused --retry-delay 3 \
    --max-time 5 "${API_URL}/health"; then
    log "Deployment details follow for troubleshooting."
    kubectl get pods,services -o wide || true
    kubectl logs deployment/heart-api --tail=80 || true
    fail "The deployment is ready, but the NodePort health check failed. Check VM networking and firewalld."
fi
printf '\n'

kubectl get pods,services -o wide
log "Deployment completed successfully."
log "Swagger UI: ${API_URL}/docs"
log "Health:     ${API_URL}/health"
log "Metrics:    ${API_URL}/metrics"
if systemctl is-active --quiet firewalld; then
    log "NOTE: firewalld is active. If another machine must reach the VM, open TCP port 30080 in your firewall after reviewing your network policy."
fi
log "Full log: ${LOG_FILE}"
