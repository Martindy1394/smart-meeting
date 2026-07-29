# Encryption at rest (near-term privacy)

Full client **end-to-end** encryption remains roadmap. This nearer-term control
encrypts **server-held audio** when `DATA_ENCRYPTION_KEY` is set.

## Enable

```bash
# Generate a key once and store it in a secrets manager:
python3 -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"

# backend/.env
DATA_ENCRYPTION_KEY=<fernet-url-safe-key>
```

Restart the API. New finalized WAVs (disk + Redis) and Redis rolling PCM are
stored as `SMENC1` + Fernet ciphertext.

## What is protected

| Asset | When key set |
|---|---|
| Finalized WAV on disk | Encrypted |
| Redis WAV cache | Encrypted |
| Redis rolling PCM window | Encrypted (get/decrypt/append/encrypt) |
| Live on-disk `.pcm` during capture | Still plaintext for append speed; deleted after finalize |
| SQLite/Postgres transcript rows | Not field-encrypted (would break keyword search) — use volume/TDE |

## Ops notes

- Losing `DATA_ENCRYPTION_KEY` makes existing ciphertext unreadable.
- Rotate by re-encrypting archives offline; do not change the key casually.
- Pair with TLS in transit and short-lived JWTs + refresh revocation
  (see auth settings).
- `/api/health` → `encryption_at_rest` shows enabled status.
