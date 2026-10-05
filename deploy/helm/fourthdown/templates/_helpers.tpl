{{- define "fourthdown.fullname" -}}
{{- if contains .Chart.Name .Release.Name -}}
{{- .Release.Name | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- printf "%s-%s" .Release.Name .Chart.Name | trunc 63 | trimSuffix "-" -}}
{{- end -}}
{{- end -}}

{{- define "fourthdown.labels" -}}
app.kubernetes.io/name: {{ .Chart.Name }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
helm.sh/chart: {{ printf "%s-%s" .Chart.Name .Chart.Version }}
{{- end -}}

{{/* Selector labels for one component: include with (dict "root" $ "component" "api"). */}}
{{- define "fourthdown.selector" -}}
app.kubernetes.io/name: {{ .root.Chart.Name }}
app.kubernetes.io/instance: {{ .root.Release.Name }}
app.kubernetes.io/component: {{ .component }}
{{- end -}}

{{- define "fourthdown.image" -}}
{{- printf "%s:%s" .repository (toString .tag) -}}
{{- end -}}

{{/* Where the app finds Ollama; empty when there is none. */}}
{{- define "fourthdown.ollamaUrl" -}}
{{- if .Values.ollama.enabled -}}
{{- printf "http://%s-ollama:11434" (include "fourthdown.fullname" .) -}}
{{- else -}}
{{- .Values.ollama.externalUrl -}}
{{- end -}}
{{- end -}}

{{- define "fourthdown.indexEnabled" -}}
{{- if and .Values.index.enabled (include "fourthdown.ollamaUrl" .) -}}true{{- end -}}
{{- end -}}

{{- define "fourthdown.postgresSecret" -}}
{{- .Values.postgres.existingSecret | default (printf "%s-postgres" (include "fourthdown.fullname" .)) -}}
{{- end -}}

{{/* Environment shared by every container running the app image. */}}
{{- define "fourthdown.appEnv" -}}
- name: PG_PASSWORD
  valueFrom:
    secretKeyRef:
      name: {{ include "fourthdown.postgresSecret" . }}
      key: password
- name: FOURTHDOWN_PG_DSN
  value: {{ printf "postgresql://%s:$(PG_PASSWORD)@%s-postgres:5432/%s" .Values.postgres.user (include "fourthdown.fullname" .) .Values.postgres.database | quote }}
{{- end -}}

{{- define "fourthdown.appSecurity" -}}
securityContext:
  runAsNonRoot: true
  runAsUser: 10001
  fsGroup: 10001
{{- end -}}

{{/* Job names carry a hash of their inputs: Job specs are immutable, so a changed input
     must be a new Job, and an unchanged one must be a no-op on upgrade. */}}
{{- define "fourthdown.pipelineJob" -}}
{{- $inputs := printf "%s|%s|%s" .Values.pipeline.seasons .Values.pipeline.trainArgs (include "fourthdown.image" .Values.images.app) -}}
{{- printf "%s-pipeline-%s" (include "fourthdown.fullname" .) ($inputs | sha256sum | trunc 8) -}}
{{- end -}}
