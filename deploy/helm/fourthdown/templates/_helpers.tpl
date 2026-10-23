{{/* Shared names and wiring for the fourthdown chart. */}}

{{/* The release-scoped name prefix every object shares. */}}
{{- define "fourthdown.name" -}}
{{- .Release.Name | trunc 40 | trimSuffix "-" -}}
{{- end -}}

{{/* Standard labels; pass a dict with "root" (the chart context) and "component". */}}
{{- define "fourthdown.labels" -}}
app.kubernetes.io/name: fourthdown
app.kubernetes.io/instance: {{ .root.Release.Name }}
app.kubernetes.io/component: {{ .component }}
{{- end -}}

{{/* Ollama's base URL: the in-cluster service, or the configured external host. */}}
{{- define "fourthdown.ollamaHost" -}}
{{- if .Values.ollama.enabled -}}
http://{{ include "fourthdown.name" . }}-ollama:11434
{{- else -}}
{{- required "ollama.externalHost is required when ollama.enabled=false" .Values.ollama.externalHost -}}
{{- end -}}
{{- end -}}

{{/* Comma-separated base URLs of the tool services, for the gateway's discovery. */}}
{{- define "fourthdown.serviceUrls" -}}
{{- $root := . -}}
{{- $urls := list -}}
{{- range $name, $svc := .Values.services -}}
{{- if ne $name "gateway" -}}
{{- $urls = append $urls (printf "http://%s-%s:%v" (include "fourthdown.name" $root) $name $svc.port) -}}
{{- end -}}
{{- end -}}
{{- join "," $urls -}}
{{- end -}}
