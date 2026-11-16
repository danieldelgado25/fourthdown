#!/usr/bin/env bash
# Start, stop, and check the demo VM from your laptop with the Azure CLI (`az login` first).
#
#   AZ_RESOURCE_GROUP=my-rg AZ_VM=my-vm deploy/azure/vm.sh start|stop|status|ip
#
# `stop` deallocates, which is what stops the hourly compute charge; a VM shut down from
# inside the OS stays allocated and keeps billing. The VM also deallocates itself when idle
# (the idle-shutdown service in docker-compose.prod.yml) and on the daily schedule that
# setup.sh creates. DRY_RUN=1 prints the az commands instead of running them.
set -euo pipefail

rg="${AZ_RESOURCE_GROUP:?set AZ_RESOURCE_GROUP to the resource group of the VM}"
vm="${AZ_VM:?set AZ_VM to the name of the VM}"

run() {
    echo "+ $*" >&2
    if [ -z "${DRY_RUN:-}" ]; then "$@"; fi
}

case "${1:-}" in
    start)
        run az vm start --resource-group "$rg" --name "$vm"
        # Every service restarts on boot; the dashboard answers once the gateway is up.
        run az vm show --resource-group "$rg" --name "$vm" --show-details \
            --query "{state: powerState, ip: publicIps, fqdn: fqdns}" --output table
        ;;
    stop)
        run az vm deallocate --resource-group "$rg" --name "$vm"
        ;;
    status)
        run az vm get-instance-view --resource-group "$rg" --name "$vm" \
            --query "instanceView.statuses[?starts_with(code, 'PowerState/')].displayStatus" \
            --output tsv
        ;;
    ip)
        run az vm show --resource-group "$rg" --name "$vm" --show-details \
            --query publicIps --output tsv
        ;;
    *)
        echo "usage: $0 start|stop|status|ip" >&2
        exit 2
        ;;
esac
