# Lancer le pipeline de recodification (CLI Argo)

- `relabel.yaml` : le workflow (étapes 0→5). Ne pas éditer pour un run normal.
- `params.yaml` : les valeurs des paramètres. C'est ce fichier qu'on édite.

Remplace `<ns>` par ton namespace (ex. `projet-ape`), ou omets `-n <ns>` si ton namespace par défaut est déjà le bon.

## 0. Prérequis (une seule fois)

Installer la CLI si `argo version` échoue (Linux x86_64, sans root) :

```bash
ARGO_VERSION=v3.6.5
curl -sLO "https://github.com/argoproj/argo-workflows/releases/download/${ARGO_VERSION}/argo-linux-amd64.gz"
gunzip argo-linux-amd64.gz
chmod +x argo-linux-amd64
mkdir -p ~/.local/bin
mv argo-linux-amd64 ~/.local/bin/argo
export PATH="$HOME/.local/bin:$PATH"
argo version
```

Sur SSP Cloud l'auth est in-cluster : pas de contexte kubectl à configurer, passe simplement `-n <ns>` à argo. Les secrets référencés par le workflow doivent exister dans le namespace (`my-s3-creds`, `hf-token`, `qdrant-apikey`, `langfuse-secrets`).

## 1. Vérifier le workflow

```bash
argo lint --offline argo-workflows/relabel.yaml
```

## 2. Lancer un run

Éditer `params.yaml`, puis :

```bash
# --watch : arbre des étapes + statuts en direct (PAS les logs)
argo submit relabel.yaml --parameter-file params.yaml -n <ns> --watch

# --log : streame les LOGS des conteneurs en direct (toutes les étapes)
argo submit relabel.yaml --parameter-file params.yaml -n <ns> --log
```

`--watch` et `--log` sont alternatifs : l'un montre l'arbre, l'autre les logs. Pour avoir les deux, soumettre avec `--watch` puis suivre les logs dans un autre terminal (`argo logs @latest -f -n <ns>`, voir §3). En fan-out multi-modèles les logs des étapes parallèles s'entremêlent. Ctrl-C coupe l'affichage mais **n'arrête pas** le run.

Surcharger une valeur sans toucher `params.yaml` (`-p` prioritaire) :

```bash
argo submit relabel.yaml --parameter-file params.yaml \
    -p mode=eval -p job-id=eval-run-01 -n <ns> --log
```

## 3. Suivre l'exécution

```bash
argo list -n <ns>                                  # tous les workflows
argo get @latest -n <ns>                           # arbre + statut du dernier run
argo logs @latest -f -n <ns>                       # logs en continu
argo logs @latest -n <ns> | grep -E "INPUT|OUTPUT" # chemins I/O
```

`@latest` = dernier workflow soumis (ou son nom exact, ex. `run-relabel-xxxxx`).

## 4. Reprendre un run qui a planté

Par défaut `job-id: ""` dans `params.yaml` → l'étape `resolve-job-id` génère un id `auto-<timestamp>` (un nouveau run à chaque soumission).

Pour **reprendre** un run planté : récupérer son id, puis le mettre dans `job-id`. Les batchs de l'étape 3 déjà écrits sur S3 sont sautés.

```bash
# 1. Retrouver le job-id du run planté (logs de l'étape resolve-job-id) :
argo logs <run-name> -n <ns> | grep "Resolved job-id"
#    (ou le lire dans un chemin OUTPUT : .../workflow_relabel/<job-id>/...)

# 2. Fixer ce job-id dans params.yaml (job-id: "auto-20260624-…") ou via -p,
#    puis relancer :
argo submit relabel.yaml --parameter-file params.yaml \
    -p job-id=auto-20260624-101530 -n <ns> --watch
```

Laisser `job-id: ""` repart toujours de zéro (nouvel id).

## 5. Arrêter / nettoyer

```bash
argo stop @latest -n <ns>        # arrêt propre (exécute les hooks de sortie)
argo terminate @latest -n <ns>   # arrêt immédiat
argo delete @latest -n <ns>      # supprimer le workflow
argo delete --completed -n <ns>  # supprimer tous les workflows terminés
```

## 6. Scénarios fréquents (valeurs dans `params.yaml`)

**Évaluation** (métriques sur lignes annotées, plusieurs modèles, vote) :
```yaml
mode: "eval"
input-url: ""                 # eval lit l'extraction filtrée aux annotées
params-conf-list: <2+ modèles>
```

**Production mono-modèle** (recode un fichier, sans vote ; l'étape ensemble tourne quand même pour écrire le fichier final) :
```yaml
mode: "prod"
input-url: "s3://projet-ape/.../mon_fichier.parquet"
params-conf-list: <1 modèle>
```

**Production multi-modèles** (recode + vote majoritaire) :
```yaml
mode: "prod"
input-url: "s3://projet-ape/.../mon_fichier.parquet"
params-conf-list: <2+ modèles>
```

**RAG avec (re)construction de la base vectorielle** :
```yaml
strategy: "rag"
build-vector-db: "true"
collection-name: "embeddings_qwen"
```

## 7. Rappel du DAG

```
0 validate ──┬─> 1 build-vector-db (rag, opt-in) ─┐
             ├─> 2 encode-unambiguous (prod) ──────┼─> 5 build-final (prod)
             └─> 3 encode (1 par modèle) ─> aggregate ─> 4 ensemble ─┘
```

- `validate` : toujours
- `build-vector-db` : si `strategy=rag` ET `build-vector-db=true`
- `encode-unambiguous` / `build-final` : prod uniquement
- `ensemble` : si plusieurs modèles, ou toujours en prod
