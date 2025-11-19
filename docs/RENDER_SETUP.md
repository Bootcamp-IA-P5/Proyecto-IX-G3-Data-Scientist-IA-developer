# Render Database Setup

## 🗄️ Database Connection

Our PostgreSQL database is hosted on **Render** (Frankfurt region).

### For Team Members

**To get database credentials:**
1. Ask the team lead for access to the Render dashboard
2. Or request the `.env` file with credentials via secure channel (Slack DM, password manager)

### Connection Details

**Host:** `dpg-d4eq9bhr0fns73bsujjg-a.frankfurt-postgres.render.com`  
**Port:** `5432`  
**Database:** `stroke_predictions_db`  
**Username:** `stroke_user`  
**Password:** *Ask team lead (not in repository)*

### Setup Steps

1. **Copy the example environment file:**
   ```bash
   cp .env.production.example .env
   ```

2. **Update `.env` with the real credentials:**
   ```env
   DATABASE_URL=postgresql://stroke_user:PASSWORD_HERE@dpg-d4eq9bhr0fns73bsujjg-a.frankfurt-postgres.render.com:5432/stroke_predictions_db
   ```

3. **Test the connection:**
   ```bash
   source venv/bin/activate
   python view_render_db.py
   ```

### Where to Find Credentials

**Option 1:** Render Dashboard (requires account access)
- Login to https://dashboard.render.com
- Navigate to: Databases → stroke-predictions-db → Connect
- Copy the "External Database URL"

**Option 2:** Team Communication
- Check team's password manager (1Password, LastPass, etc.)
- Ask in the team Slack channel: #stroke-prediction-dev
- Contact: [team lead email/slack]

### Security Notes

⚠️ **NEVER commit the real `.env` file to git**  
⚠️ **Don't share credentials in public channels**  
✅ Use secure channels (encrypted messages, password managers)  
✅ Rotate credentials if exposed

### Verify Connection

Once you have credentials, verify with:

```bash
python -c "from backend.database.connection import init_db; init_db(); print('✅ Connected!')"
```

You should see: `✅ Connected!`

---

**Need help?** Contact the DevOps team or check the main [README_DB.md](./README_DB.md)
