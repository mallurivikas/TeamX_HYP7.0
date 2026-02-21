# WhatsApp Integration Setup Guide

## Prerequisites
1. Twilio account (free trial available)
2. WhatsApp Business API access through Twilio
3. Public HTTPS URL for webhooks (use ngrok for development)

## Step 1: Twilio Setup

1. **Create Twilio Account**
   - Go to https://www.twilio.com/try-twilio
   - Sign up for free account

2. **Get Twilio Sandbox for WhatsApp**
   - Navigate to Console > Messaging > Try it out > Send a WhatsApp message
   - Join sandbox by sending code to provided WhatsApp number
   - Note your Sandbox number (e.g., +1 415 523 8886)

3. **Get Credentials**
   - Account SID: Console > Account Info
   - Auth Token: Console > Account Info
   - WhatsApp Number: Console > Messaging > WhatsApp Sandbox

## Step 2: Environment Configuration

1. Copy `.env.example` to `.env`
2. Fill in your Twilio credentials:
   ```
   TWILIO_ACCOUNT_SID=ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
   TWILIO_AUTH_TOKEN=your_auth_token_here
   TWILIO_WHATSAPP_NUMBER=whatsapp:+14155238886
   ```

## Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

## Step 4: Setup Webhook URL

### For Development (using ngrok):

1. Install ngrok: https://ngrok.com/download
2. Start your Flask app: `python app.py`
3. In new terminal, run: `ngrok http 5000`
4. Copy the HTTPS URL (e.g., https://c8b291dce5de.ngrok-free.app)
5. Your webhook URL will be: `https://c8b291dce5de.ngrok-free.app/whatsapp/webhook`
6. Go to Twilio Console and paste this URL in "When a message comes in"

### Configure Twilio Webhook:

1. Go to Twilio Console > Messaging > Settings > WhatsApp Sandbox
2. In the "When a message comes in" field, enter: `https://c8b291dce5de.ngrok-free.app/whatsapp/webhook`
3. Method: **POST**
4. Click "Save" button at the bottom
5. Your webhook is now configured!

**Important Notes:**
- The URL must include `/whatsapp/webhook` at the end
- Make sure your Flask app is running before testing
- ngrok URL changes each time you restart ngrok (unless you have a paid account)
- If ngrok restarts, you'll need to update this URL in Twilio Console

## Step 5: Test Integration

1. Send "Hi" to your Twilio WhatsApp sandbox number
2. You should receive a welcome message
3. Send a photo of nutrition label
4. Receive instant analysis results

## Production Deployment

### For Production:

1. Deploy app to cloud platform (Heroku, AWS, Google Cloud)
2. Get permanent HTTPS URL
3. Update webhook in Twilio Console
4. Apply for WhatsApp Business API (if needed)

### Heroku Deployment:

```bash
# Create Procfile
echo "web: python app.py" > Procfile

# Deploy
heroku create your-app-name
heroku config:set TWILIO_ACCOUNT_SID=your_sid
heroku config:set TWILIO_AUTH_TOKEN=your_token
heroku config:set TWILIO_WHATSAPP_NUMBER=whatsapp:+14155238886
git push heroku main
```

## Quick Setup Checklist:
- [x] Twilio Account SID: AC271014c53ac57e81101f47a7fb25094e
- [x] Twilio Auth Token: Configured
- [x] WhatsApp Number: whatsapp:+14155238886
- [ ] Flask app running on port 5000
- [ ] ngrok running and forwarding to localhost:5000
- [x] Webhook URL in Twilio: `https://c8b291dce5de.ngrok-free.app/whatsapp/webhook`
- [ ] Method set to POST in Twilio
- [ ] Saved configuration in Twilio Console

## Quick Test Commands

Once configured, users can send:

1. **Text Messages:**
   - `hi` or `hello` → Welcome message with instructions
   - `scan` → How to scan nutrition labels
   - `assess` → Link to full health assessment
   - `help` → Show all available commands

2. **Photo Messages:**
   - Send any photo of a nutrition label → Get instant AI analysis
   - Receive personalized recommendations based on health profile

## Verify Integration

1. **Check endpoint status:**
   ```bash
   curl https://c8b291dce5de.ngrok-free.app/whatsapp/test
   ```

2. **Expected response:**
   ```json
   {
     "status": "WhatsApp integration active",
     "twilio_client": "Connected ✅",
     "webhook_url": "https://c8b291dce5de.ngrok-free.app/whatsapp/webhook"
   }
   ```

3. **Test via WhatsApp:**
   - Join Twilio sandbox (send join code to sandbox number)
   - Send "hi" to the sandbox number
   - Should receive welcome message

## Troubleshooting Steps

**Issue: "Twilio client not initialized"**
- Check credentials are correctly hardcoded
- Verify internet connection
- Check Twilio account status

**Issue: "Webhook not receiving messages"**
- Verify ngrok is running: `ngrok http 5000`
- Check webhook URL includes `/whatsapp/webhook`
- Ensure Flask app is running
- Check ngrok web interface: http://127.0.0.1:4040

**Issue: "Image not processing"**
- Verify `/api/analyze-nutrition` endpoint exists in app.py
- Check Flask app logs for errors
- Ensure image is clear and well-lit

**Issue: "Connection timeout"**
- Verify Flask app is accessible from internet
- Check firewall settings
- Try restarting ngrok with: `ngrok http 5000`

## Cost Considerations

- Twilio Trial: Free credits for testing
- Production: ~$0.005 per message (WhatsApp)
- For high volume, consider upgrading plan

## Security Notes

- Never commit .env file to version control
- Validate Twilio request signatures in production
- Implement rate limiting for API endpoints
- Secure webhook endpoints

## Support

For issues or questions:
- Twilio Docs: https://www.twilio.com/docs/whatsapp
- GitHub Issues: [Your repo URL]
