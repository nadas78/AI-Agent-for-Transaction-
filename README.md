## Dataset et Préparation des Données

Le projet utilise un **dataset synthétique de transactions DeFi** généré en langage naturel pour entraîner le modèle AI à diagnostiquer les erreurs des transactions.

### Génération des données

- **Nombre de lignes** : 5000 samples.
- **Contenu** : chaque sample contient un input en langage naturel décrivant une transaction échouée et un output structuré comprenant :
  - `Technical Explanation` : explication technique de l’erreur.
  - `Simplified Explanation` : version simple compréhensible par un utilisateur.
  - `Severity` : niveau de gravité (`Low`, `Medium`, `High`).
  - `Recommended Fix` : recommandation pour corriger l’erreur.
  - `Confidence` : niveau de confiance du modèle sur la prédiction.

- **Variables simulées** :
  - `networks` : Ethereum, Polygon, BSC
  - `transaction_types` : swap, stake, borrow, repay, add_liquidity, remove_liquidity
  - `protocols` : Uniswap, Aave, PancakeSwap, Curve, Compound
  - `tokens` : ETH, USDT, USDC, DAI, WBTC, MATIC, BNB
  - Autres paramètres : `gas_limit`, `gas_used`, `gas_price`, `slippage`, `balance`, `allowance`, `deadline`, `nonce_issue`, `contract_paused`, `overflow_flag`, `congestion`.

### Nettoyage et Préparation

- Les données générées sont divisées en **train (80%)**, **validation (10%)** et **test (10%)**.
- Chaque input/output est combiné et tokenisé pour être compatible avec le modèle GPT-Neo.
- Les labels sont préparés pour le calcul de la loss lors du fine-tuning.
- Les fichiers temporaires et inutiles (ex: `__pycache__`) sont ignorés grâce à un `.gitignore`.

### Modèle Utilisé

- **Base model** : `EleutherAI/gpt-neo-125M`
- **Fine-tuning** : LoRA (`Low-Rank Adaptation`) pour un entraînement léger et efficace sur le dataset DeFi.
- **Tokenizer** : AutoTokenizer compatible GPT-Neo.
- **Paramètres LoRA** :
  - r = 8, lora_alpha = 32
  - target_modules = ["c_attn", "c_proj"]
  - lora_dropout = 0.1
  - task_type = "CAUSAL_LM"

### Entraînement et Évaluation

- **Batch size** : 4
- **Epochs** : 5 (réglable selon besoins)
- **Tokenization** : max_length = 256
- **Loss** : Cross-Entropy pour génération de texte
- **Évaluation** :
  - Calcul de la **perplexité** sur validation et test
  - Export du modèle fine-tuné et du tokenizer pour intégration backend

### Sortie

- Modèle exporté pour FastAPI backend : `./exported_model`
  
## Fonctionnalités

- **Analyse des transactions DeFi** : détecte les erreurs courantes comme `insufficient funds`, `out of gas`, `nonce too low`, `contract paused`, et plus.
- **Explications techniques et simplifiées** : fournit à la fois une description technique pour les développeurs et une explication simple pour les utilisateurs.
- **Recommandations de correction** : propose des solutions concrètes pour résoudre les erreurs et réussir la transaction.
- **Support multi-chain** : fonctionne avec Ethereum, Polygon, Binance Smart Chain et les protocoles populaires tels que Uniswap, Aave, PancakeSwap, Curve, Compound.
- **Interface API avec FastAPI** : permet de facilement intégrer l'agent dans des applications web ou des interfaces front-end.
- **Modèle AI basé sur GPT-Neo et LoRA** : fine-tuné sur des exemples de transactions DeFi pour fournir des diagnostics précis.

## Technologies utilisées

- Python 3.10+
- [FastAPI](https://fastapi.tiangolo.com/) pour le backend API
- [Transformers](https://huggingface.co/docs/transformers/index) et [LoRA](https://github.com/huggingface/peft) pour le modèle NLP
- Hugging Face Hub pour le stockage et la gestion du modèle AI

- LoRA adapter léger : `./lora_adapter`
- Dataset sauvegardé en JSON pour reproduire ou améliorer l’entraînement
- Zip disponible pour téléchargement depuis Colab ou serveur
