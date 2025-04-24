# Comet ML


Comet’s machine learning platform integrates with your existing infrastructure and tools so you can manage, visualize, and optimize models from training runs to production monitoring. This plugin integrates Flyte with Comet by configuring links between the two platforms.

To install the plugin, run:

```
$ pip install flytekitplugins-comet-ml
```

Comet requires an API key to authenticate with their platform. In the example, a secret is created using
[Flyte's Secrets manager](https://www.union.ai/docs/flyte/deployment/flyte-configuration/secrets/#creating-secrets-with-a-secrets-manager).

To enable linking from the Flyte side panel to Comet.ml, add the following to Flyte's configuration:

```yaml
plugins:
  logs:
    dynamic-log-links:
      - comet-ml-execution-id:
          displayName: Comet
          templateUris: "{{ .taskConfig.host }}/{{ .taskConfig.workspace }}/{{ .taskConfig.project_name }}/{{ .executionName }}{{ .nodeId }}{{ .taskRetryAttempt }}{{ .taskConfig.link_suffix }}"
      - comet-ml-custom-id:
          displayName: Comet
          templateUris: "{{ .taskConfig.host }}/{{ .taskConfig.workspace }}/{{ .taskConfig.project_name }}/{{ .taskConfig.experiment_key }}"
```
