# hackathon
This will be the Repository for the First Hackathon attended by Team "Last Brain Cells" .
🛡️ CyberSecure: AI-Powered Real-Time Network Intrusion Triage
Team: Last Brain Cells | Event: Redact Hackathon 2025

🔮 Vision
To revolutionize SOC operations by creating an autonomous, tamper-proof first line of defense that eliminates alert fatigue and neutralizes cyber threats in milliseconds before they cause harm.

🚀 The Problem: Alert Fatigue
Security Operations Centers (SOCs) are drowning. Analysts receive over 10,000 security alerts daily, with 90% being false positives.

The Consequence: Critical threats (like Ransomware or DDoS) slip through the noise.

The Gap: Existing tools just "alert." They don't "decide."

💡 Our Solution
CyberSecure is not just an IDS (Intrusion Detection System); it is an Automated Triage System.

Detects: Classifies traffic as Benign vs. Specific Attacks (DoS, Probe, etc.) using 99.83% Recall AI.

Decides: Assigns a Confidence Score and maps it to a concrete Security Action (e.g., "Block IP").

Secures: Hashes every decision onto an immutable Blockchain Ledger for auditability.

🛠️ Key Features (MVP + Bonuses)
1. 🧠 AI Core (High Recall)
Trained on NSL-KDD Dataset (500k+ Records).

Multi-Class Classification: Distinguishes between Normal, DoS, Probe, U2R, and R2L attacks.

Performance: Achieved 99.83% Accuracy and 99.83% Weighted Recall on the validation set.

2. ⛓️ Blockchain Audit Trail (Bonus)
Every log entry is cryptographically hashed using SHA-256.

Each block contains the hash of the previous block.

Result: Logs cannot be deleted or altered by insider threats without breaking the chain.

3. 🔍 Explainable AI (XAI) (Bonus)
We don't trust "Black Box" AI.

Integrated Feature Importance Analysis to visualize exactly why the model flagged a packet (e.g., High Duration, Source Bytes).

4. 🛡️ Mission Control Dashboard
Dark Mode UI: Optimized for SOC environments.

Live Threat Graph: Real-time visualization of attack probability.

Threat Vault: A dedicated archive for critical incidents.

Data Lab: Upload external CSVs to scan them offline.

⚙️ Technology Stack
Language: Python 3.9+

Frontend: Streamlit (Custom CSS for Enterprise UI)

Machine Learning: Scikit-Learn (Random Forest Classifier), Pandas, NumPy

Security: Hashlib (SHA-256 implementation)

Visualization: Matplotlib, Streamlit Charts