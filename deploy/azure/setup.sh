#!/usr/bin/env bash
# One-time setup for the demo VM's automatic shutdowns. Run from your laptop with the Azure
# CLI (`az login` first); nothing in CI or on the VM runs it.
#
#   AZ_RESOURCE_GROUP=my-rg AZ_VM=my-vm deploy/azure/setup.sh [--time 0500] [--email you@example.com]
#
# It sets up two independent shutdowns, both of which deallocate (stop billing):
#
# 1. A daily schedule, the CLI form of the portal's "Auto-shutdown" toggle. --time is UTC,
#    HHMM (default 0500, about midnight US Central); --email gets a warning 30 minutes
#    before.
# 2. Permission for the idle-shutdown service (fourthdown.serving.idle) to deallocate the VM
#    after an idle hour: a system-assigned managed identity on the VM, given the built-in
#    "Virtual Machine Contributor" role on this one VM and nothing else. The service then
#    gets its token from the instance metadata service; no secret is stored anywhere.
#
# Safe to rerun. DRY_RUN=1 prints the az commands instead of running them.
set -euo pipefail

rg="${AZ_RESOURCE_GROUP:?set AZ_RESOURCE_GROUP to the resource group of the VM}"
vm="${AZ_VM:?set AZ_VM to the name of the VM}"
time_utc="0500"
email=""

while [ $# -gt 0 ]; do
    case "$1" in
        --time) time_utc="$2"; shift 2 ;;
        --email) email="$2"; shift 2 ;;
        *) echo "unknown option $1" >&2; exit 2 ;;
    esac
done

run() {
    echo "+ $*" >&2
    if [ -z "${DRY_RUN:-}" ]; then "$@"; fi
}

echo "1/2 daily auto-shutdown at ${time_utc} UTC" >&2
schedule=(az vm auto-shutdown --resource-group "$rg" --name "$vm" --time "$time_utc")
if [ -n "$email" ]; then schedule+=(--email "$email"); fi
run "${schedule[@]}"

echo "2/2 managed identity allowed to deallocate this VM" >&2
if [ -n "${DRY_RUN:-}" ]; then
    principal="<principal-id>"
    scope="<vm-resource-id>"
else
    principal="$(az vm identity assign --resource-group "$rg" --name "$vm" \
        --query systemAssignedIdentity --output tsv)"
    scope="$(az vm show --resource-group "$rg" --name "$vm" --query id --output tsv)"
fi
echo "+ az vm identity assign --resource-group $rg --name $vm" >&2
run az role assignment create --assignee-object-id "$principal" \
    --assignee-principal-type ServicePrincipal \
    --role "Virtual Machine Contributor" --scope "$scope"

echo "done. On the VM, the idle-shutdown service picks this up on its next check." >&2
