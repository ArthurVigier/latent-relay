# CLAUDE CODE — ERIS v5 Platonic Cognitive Bridge
# System Prompt pour Phase 0 : Extension du Latent Relay en Zombie Bridge

## CONTEXTE DU PROJET

Tu travailles sur le repo https://github.com/ArthurVigier/latent-relay
C'est un wrapper MCP/FastAPI autour de LatentMAS (Zou et al., 2025) qui permet à des agents IA de raisonner dans l'espace latent via transfert de KV-cache.

L'objectif est d'ÉTENDRE ce repo pour construire ERIS v5 — un "Platonic Cognitive Bridge" qui crée un canal de communication latent entre Claude (closed-source, communique en texte uniquement), un modèle open-source "zombie" (Qwen3-14B/32B ou DeepSeek-V3, hidden states entièrement accessibles), et un humain (Arthur, qui intervient en PyTorch direct).

Le zombie n'est PAS un agent autonome. C'est un ESPACE COGNITIF PARTAGÉ — un modèle dont les représentations internes sont inspectables et manipulables, et qui sert de traducteur bidirectionnel texte ↔ latent entre les participants.

## CODE EXISTANT À CONNAÎTRE

### Structure du repo
```
latent-relay/
├── engine.py              # CORE — LatentRelayEngine : sessions, W_a, think(), collaborate()
├── server.py              # FastAPI REST server (wraps engine)
├── mcp_server.py          # MCP tool server (wraps engine)
├── test_e2e.py            # End-to-end test
├── requirements.txt
├── openclaw_compat/
│   ├── openai_proxy.py    # OpenAI-compatible proxy pour OpenClaw
│   └── VALIDATION_LOG.md
├── patches/               # DeepSeek MLA adapter
└── phase0/                # Feasibility validation suite
```

### engine.py — À comprendre en détail

L'engine expose 3 opérations :
1. `create_session()` → charge le modèle, calcule W_a (alignment matrix), retourne session_id
2. `think(session_id, prompt, n_steps, role, inherit_from)` → encode le prompt, fait N passes de latent rollout (hidden→W_a→forward→hidden), stocke le KV-cache sous un handle, retourne le handle + métadonnées
3. `collaborate(session_id, handles, final_prompt)` → génère du texte à partir du contexte latent accumulé

Concepts clés dans le code :
- `W_a` : matrice d'alignement calculée par ridge regression entre output embeddings et input embeddings. Permet de réinjecter un hidden state comme input embedding. Calculée une seule fois au startup.
- `_apply_realignment(hidden)` : applique W_a + normalisation. C'est le cœur du latent rollout.
- `LatentThought` : dataclass stockant handle, KV-cache, hidden_embedding, metadata.
- `Session` : dataclass stockant les thoughts d'une session.
- Le latent rollout dans `think()` boucle N fois : `hidden → W_a → model(inputs_embeds=latent) → new_hidden`.
- `hidden_embedding` est le DERNIER hidden state (couche finale) du modèle après chaque passe. Il est stocké dans le thought mais N'EST PAS exposé via l'API actuelle — c'est un des points à étendre.

### server.py — FastAPI existant

Endpoints actuels :
- POST /sessions, GET /sessions, DELETE /sessions/{id}
- POST /think, POST /collaborate, GET /thoughts/{session_id}/{handle}

### Phase 0 validation (déjà faite)
- KV-cache serialization : 0.099s ✓
- int8 quantization : cosine 0.9993 ✓
- W_a : bit-identical, spectral radius 1.0 ✓
- Le latent rollout est stable sur 60 steps ✓

## CE QU'IL FAUT CONSTRUIRE

### Objectif : étendre engine.py et server.py pour supporter le "Zombie Bridge"

Il faut ajouter 5 capacités au système existant SANS casser les fonctionnalités actuelles (LatentMAS, MCP, OpenClaw compat doivent continuer de fonctionner) :

### 1. Endpoint `/v1/encode` — Encoder du texte et exposer les hidden states

```python
# INPUT
{
    "text": "texte de Claude à encoder",
    "return_layers": [15, 20, 25, -1],  # quelles couches retourner (-1 = dernière)
    "return_attention": false,            # optionnel : poids d'attention
    "session_id": "optional"              # si fourni, stocke dans la session
}

# OUTPUT
{
    "handle": "enc_abc123",               # handle pour référencer plus tard
    "hidden_states": {
        "layer_15": [...],                # liste de floats, shape [seq_len, hidden_dim]
        "layer_20": [...],
        "layer_25": [...],
        "last": [...]
    },
    "tokens": ["token1", "token2", ...],  # les tokens produits par le tokenizer
    "seq_len": 42,
    "hidden_dim": 3584                    # pour Qwen3-14B
}
```

IMPORTANT : les hidden states doivent être retournés en float32 (pas bfloat16) et base64-encodés pour le transport JSON. Fournir un helper `decode_hidden_states(response)` côté client.

Implémentation dans engine.py : nouveau méthode `encode()` qui fait un forward pass avec `output_hidden_states=True` et retourne les couches demandées. Le résultat est stocké comme un `LatentThought` avec le KV-cache ET les hidden states.

### 2. Endpoint `/v1/analyze` — Analyse MI des hidden states

```python
# INPUT
{
    "handle": "enc_abc123",
    "analyses": ["sae_features", "a_hat", "cosine_map", "pca_3d", "token_norms"]
}

# OUTPUT
{
    "sae_features": {
        "top_20": [
            {"index": 4521, "activation": 3.72, "label": "code_architecture"},  # si labels dispo
            ...
        ]
    },
    "a_hat_score": 0.73,                 # score d'agentivité de la sonde Â-hat
    "cosine_map": {                      # similarité cosinus avec des concepts de référence
        "game_theory": 0.34,
        "thermodynamics": 0.12,
        "software_architecture": 0.67
    },
    "pca_3d": [[x, y, z], ...],         # projection PCA 3D des hidden states par token
    "token_norms": [12.3, 11.8, ...]     # norme de chaque token dans le hidden space
}
```

Implémentation : nouveau module `eris/analyzers.py` avec des classes pluggables :
- `SAEAnalyzer` : charge une SAE pré-entraînée (sae-lens format), forward les hidden states, retourne top-k features. La SAE est chargée une seule fois.
- `AHatAnalyzer` : charge la sonde Â-hat (a_hat_optimizer format), calcule le score d'agentivité. La sonde est chargée une seule fois.
- `CosineMapAnalyzer` : maintient un dictionnaire de "concept vectors" de référence (pré-calculés). Calcule la similarité cosinus entre le hidden state moyen et chaque concept.
- `PCAAnalyzer` : PCA incrémentale (sklearn) sur les hidden states, retourne la projection 3D.
- `NormAnalyzer` : trivial, norme L2 par token.

Les analyzers sont OPTIONNELS — si la SAE n'est pas chargée, l'analyse SAE retourne null. L'API ne crashe pas si un analyzer manque.

### 3. Endpoint `/v1/latent_think` étendu — Retourner la trajectoire

L'endpoint `think` actuel fait le latent rollout mais ne retourne que le handle final. Il faut l'étendre pour OPTIONNELLEMENT retourner la trajectoire complète des hidden states à chaque passe.

```python
# INPUT (extension du think existant)
{
    "session_id": "...",
    "prompt": "texte de Claude",
    "n_steps": 60,
    "return_trajectory": true,           # NOUVEAU : retourner les hidden states intermédiaires
    "trajectory_analyses": ["a_hat", "norms"],  # NOUVEAU : analyses à chaque passe
    "perturbation": null                  # NOUVEAU : vecteur de perturbation (cf. point 4)
}

# OUTPUT (extension du think existant)
{
    "handle": "t_...",
    "n_steps": 60,
    "elapsed_s": 2.87,
    "trajectory": [                      # NOUVEAU
        {"step": 0, "hidden_norm": 12.3, "a_hat": 0.72, "displacement": 0.0},
        {"step": 1, "hidden_norm": 12.5, "a_hat": 0.68, "displacement": 0.34},
        ...
        {"step": 59, "hidden_norm": 14.1, "a_hat": 0.81, "displacement": 3.47}
    ],
    "total_displacement": 3.47           # NOUVEAU : distance z_0 → z_K
}
```

Implémentation dans engine.py : modifier la boucle de latent rollout dans `think()` pour :
- Stocker `last_hidden` à chaque step si `return_trajectory=True`
- Calculer `displacement = cosine_distance(z_0, z_k)` à chaque step
- Appliquer les analyzers demandés à chaque step (attention au coût — SAE à chaque step est cher, a_hat est cheap)
- Appliquer la perturbation si fournie (cf. point 4)

### 4. Endpoint `/v1/inject` — Intervention chirurgicale

```python
# INPUT
{
    "handle": "t_abc123",                # thought à modifier
    "operation": "add",                  # "add", "steer", "replace"
    "layer": -1,                         # couche (-1 = dernière, ou index spécifique)
    "position": -1,                      # position dans la séquence (-1 = dernier token)
    "vector": [...],                     # vecteur à injecter (float32, même dim que hidden)
    "scale": 0.5                         # facteur de scaling
}

# OUTPUT
{
    "status": "injected",
    "old_norm": 12.3,
    "new_norm": 13.1,
    "cosine_shift": 0.12                 # cosinus entre ancien et nouveau hidden state
}
```

Implémentation : ATTENTION — ceci modifie le hidden_embedding stocké dans le thought, PAS le KV-cache directement. L'injection modifie le point de départ pour le PROCHAIN latent rollout. Si on veut que l'injection persiste dans la génération, il faut :
- Soit faire un nouveau `think()` avec `inherit_from=[handle_injecté]` et 0 steps (pour propager l'injection dans le KV-cache)
- Soit modifier directement le KV-cache (plus complexe, Phase 2)

Pour Phase 0, l'injection sur le hidden_embedding + re-think est suffisante.

### 5. Endpoint `/v1/bridge` — La boucle complète Claude → Zombie → Claude

C'est le point d'entrée principal pour ERIS. Il orchestre la boucle complète :

```python
# INPUT
{
    "claude_text": "le texte que Claude a généré",
    "mode": "passive",                   # "passive", "ruminate", "analyze_only"
    "n_steps": 60,                       # si mode=ruminate
    "analyses": ["sae_features", "a_hat"],
    "decode_after": true,                # générer du texte après rumination
    "max_new_tokens": 512
}

# OUTPUT
{
    "enriched_text": "texte généré par le zombie après rumination",
    "analysis": {
        "sae_features": {...},
        "a_hat_score": 0.73,
        "implicit_features": [           # features activées en latent mais non présentes dans le texte
            {"index": 4521, "label": "hidden_dependency", "activation": 2.1}
        ]
    },
    "trajectory_summary": {
        "total_displacement": 3.47,
        "max_a_hat": 0.81,
        "steps_to_convergence": 42
    }
}
```

Implémentation : c'est un pipeline qui chaîne encode → analyze → latent_think → analyze → decode. Le champ `implicit_features` est calculé en comparant les features SAE activées dans les hidden states avec les tokens du texte d'entrée — une feature est "implicite" si elle est activée mais que son label ne correspond à aucun token ou bigramme du texte original.

## STRUCTURE CIBLE DU REPO

```
latent-relay/
├── engine.py                    # EXISTANT — NE PAS CASSER
├── server.py                    # EXISTANT — ÉTENDRE avec nouveaux endpoints
├── mcp_server.py                # EXISTANT — NE PAS TOUCHER (Phase 0)
├── test_e2e.py                  # EXISTANT
├── requirements.txt             # ÉTENDRE (ajouter sae-lens, sentence-transformers, etc.)
├── openclaw_compat/             # EXISTANT
├── patches/                     # EXISTANT
├── phase0/                      # EXISTANT
│
├── eris/                        # NOUVEAU — tout le code ERIS v5
│   ├── __init__.py
│   ├── bridge.py                # Pipeline bridge : encode → analyze → think → decode
│   ├── analyzers.py             # SAEAnalyzer, AHatAnalyzer, CosineMapAnalyzer, etc.
│   ├── injector.py              # Injection de vecteurs dans les hidden states
│   ├── trajectory.py            # Tracking de trajectoire pendant le latent rollout
│   ├── implicit_features.py     # Détection de features implicites (SAE vs texte)
│   └── config.py                # Config ERIS (chemins SAE, sonde Â-hat, concepts de réf.)
│
├── eris_server.py               # NOUVEAU — FastAPI server étendant server.py
│                                #   Importe tout de server.py + ajoute /v1/encode,
│                                #   /v1/analyze, /v1/inject, /v1/bridge
│
├── eris_client.py               # NOUVEAU — Client Python pour la boucle Claude→zombie→Claude
│                                #   Utilise anthropic SDK + httpx vers eris_server
│
├── test_bridge.py               # NOUVEAU — Tests du canal
│
└── configs/
    ├── eris_config.yaml          # NOUVEAU — config paths SAE, Â-hat, etc.
    └── concept_vectors/          # NOUVEAU — vecteurs de concepts de référence pré-calculés
```

## CONTRAINTES TECHNIQUES

1. **NE PAS casser l'existant.** Les tests `test_e2e.py` doivent continuer de passer. Le MCP server et le OpenClaw proxy ne sont pas touchés en Phase 0.

2. **Le serveur ERIS est un SUPERSET du serveur existant.** `eris_server.py` importe `server.py` et ajoute les endpoints. On peut lancer l'un ou l'autre.

3. **Les analyzers sont optionnels et lazy-loaded.** Si la SAE n'est pas configurée, les endpoints retournent `null` pour les analyses SAE. Pas de crash, pas de dépendances obligatoires.

4. **Les hidden states en JSON.** Les tenseurs sont convertis en float32, puis en listes Python (ou base64 pour les gros tenseurs). Un flag `compact=true` dans la requête utilise base64, `compact=false` (défaut) utilise des listes lisibles.

5. **Thread safety.** L'engine existant utilise un `threading.Lock()`. Les nouvelles opérations doivent respecter ce lock.

6. **GPU memory.** Les hidden states de trajectoire sont GROS. Pour un modèle 14B avec 60 steps, chaque snapshot de hidden state fait ~14KB (3584 × float32). 60 steps × 14KB = ~840KB par trajectoire. C'est gérable mais il ne faut pas stocker en GPU — copier en CPU dès le calcul.

7. **Modèles supportés.** Le code doit fonctionner sur :
   - Qwen3-4B, 14B, 32B (principal)
   - DeepSeek-V2-Lite (avec le patch MLA existant dans patches/)
   - Tout modèle HuggingFace avec `output_hidden_states=True` et standard KV-cache

## ORDRE D'IMPLÉMENTATION RECOMMANDÉ

1. `eris/config.py` — Parsing de config, chemins des modèles SAE/Â-hat
2. `eris/trajectory.py` — Tracking de trajectoire (liste de hidden states + métriques par step)
3. Modification de `engine.py` — Ajouter `encode()` et étendre `think()` pour trajectoire + perturbation
4. `eris/analyzers.py` — SAE, Â-hat, cosine map, PCA, norms
5. `eris/injector.py` — Injection de vecteurs dans hidden_embedding
6. `eris/implicit_features.py` — Comparaison features SAE vs tokens du texte
7. `eris/bridge.py` — Pipeline complet encode→analyze→think→decode
8. `eris_server.py` — FastAPI server avec tous les endpoints
9. `eris_client.py` — Client pour la boucle Claude→zombie→Claude
10. `test_bridge.py` — Tests

## BIBLIOTHÈQUES

```
# Existant
torch
transformers
fastapi
uvicorn
pydantic

# À ajouter
sae-lens              # SAEs pré-entraînées (optionnel, lazy-load)
a-hat-optimizer       # Sonde Â-hat d'Arthur (optionnel, lazy-load)
sentence-transformers # Pour la détection de features implicites
scikit-learn          # PCA, clustering
numpy                 # Calculs
anthropic             # Client Claude pour eris_client.py
httpx                 # Client HTTP async
pyyaml                # Config
```

## NOTES ARTHUR

- Le code doit être propre, bien documenté, avec des docstrings.
- Chaque endpoint a un exemple curl dans la docstring.
- Les erreurs sont descriptives (pas juste 500 Internal Server Error).
- Le code suit le style du engine.py existant (classes avec dataclasses, pas de magie).
- Les tests sont écrits en même temps que le code, pas après.
- Si un choix d'implémentation n'est pas clair, demande — ne suppose pas.
