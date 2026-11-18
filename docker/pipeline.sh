#!/bin/sh
# Build the warehouse and train the models into $FOURTHDOWN_DATA_DIR, skipping the work
# when the volume already holds a verified build of the same configuration. This is the
# compose `pipeline` service and the Helm pipeline Job.
set -eu

data="${FOURTHDOWN_DATA_DIR:?FOURTHDOWN_DATA_DIR must be set}"
seasons="${FOURTHDOWN_SEASONS:-2009-2025}"
train_args="${FOURTHDOWN_TRAIN_ARGS:-}"
want="seasons=${seasons} train=${train_args}"
stamp="${data}/.pipeline"

# The warehouse service's miss log: the one directory a serving container writes. Made
# here, as the data's owner, so a bind mount or subPath finds it writable.
mkdir -p "${data}/feedback"

if [ -f "${data}/models/model_card.json" ] && [ "$(cat "${stamp}" 2>/dev/null)" = "${want}" ] \
    && fourthdown lineage --verify --output /tmp/lineage.md; then
    # Rebuild the dbt models anyway: cheap next to ingest and training (seconds), and the
    # data tests run again against whatever the image's dbt project now says.
    fourthdown warehouse
    echo "data and models are current (${want}); views refreshed"
    exit 0
fi

mkdir -p "${data}/reports"
fourthdown build --seasons "${seasons}"
# shellcheck disable=SC2086 # train_args is a list of flags
fourthdown train --output "${data}/reports/model_eval.md" ${train_args}
fourthdown lineage --output "${data}/reports/lineage.md"
echo "${want}" > "${stamp}"
