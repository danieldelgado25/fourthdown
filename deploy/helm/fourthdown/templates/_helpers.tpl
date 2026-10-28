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

{{/* Selector labels for one component: include with (dict "root" $ "component" "gateway"). */}}
{{- define "fourthdown.selector" -}}
app.kubernetes.io/name: {{ .root.Chart.Name }}
app.kubernetes.io/instance: {{ .root.Release.Name }}
app.kubernetes.io/component: {{ .component }}
{{- end -}}

{{- define "fourthdown.image" -}}
{{- printf "%s:%s" .repository (toString .tag) -}}
{{- end -}}

{{/* An image built from the repo Dockerfile: include with (dict "root" $ "target" "models"). */}}
{{- define "fourthdown.appImage" -}}
{{- printf "%s-%s:%s" .root.Values.images.prefix .target (toString .root.Values.images.tag) -}}
{{- end -}}

{{/* Where the app finds Ollama; empty when there is none. */}}
{{- define "fourthdown.ollamaUrl" -}}
{{- if .Values.ollama.enabled -}}
{{- printf "http://%s-ollama:11434" (include "fourthdown.fullname" .) -}}
{{- else -}}
{{- .Values.ollama.externalUrl -}}
{{- end -}}
{{- end -}}

{{/* The chat endpoint: llm.baseUrl, else Ollama's OpenAI-compatible root; empty when neither. */}}
{{- define "fourthdown.llmBaseUrl" -}}
{{- if .Values.llm.baseUrl -}}
{{- .Values.llm.baseUrl -}}
{{- else if include "fourthdown.ollamaUrl" . -}}
{{- printf "%s/v1" (include "fourthdown.ollamaUrl" .) -}}
{{- end -}}
{{- end -}}

{{- define "fourthdown.indexEnabled" -}}
{{- if and .Values.index.enabled (include "fourthdown.ollamaUrl" .) -}}true{{- end -}}
{{- end -}}

{{- define "fourthdown.secret" -}}
{{- printf "%s-secrets" (include "fourthdown.fullname" .) -}}
{{- end -}}

{{/* Comma-separated base URLs of the tool services, for the gateway's discovery. */}}
{{- define "fourthdown.serviceUrls" -}}
{{- $root := . -}}
{{- $urls := list -}}
{{- range $name, $svc := .Values.services -}}
{{- if ne $name "gateway" -}}
{{- $urls = append $urls (printf "http://%s-%s:%v" (include "fourthdown.fullname" $root) $name $svc.port) -}}
{{- end -}}
{{- end -}}
{{- join "," $urls -}}
{{- end -}}

{{/* Credentials every app container may need, from the release Secret. */}}
{{- define "fourthdown.secretEnv" -}}
- name: FOURTHDOWN_PG_DSN
  valueFrom:
    secretKeyRef:
      name: {{ include "fourthdown.secret" . }}
      key: dsn
- name: FOURTHDOWN_LLM_API_KEY
  valueFrom:
    secretKeyRef:
      name: {{ include "fourthdown.secret" . }}
      key: llmApiKey
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
{{- $inputs := printf "%s|%s|%s" .Values.pipeline.seasons .Values.pipeline.trainArgs (include "fourthdown.appImage" (dict "root" . "target" "pipeline")) -}}
{{- printf "%s-pipeline-%s" (include "fourthdown.fullname" .) ($inputs | sha256sum | trunc 8) -}}
{{- end -}}
