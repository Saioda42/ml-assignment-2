# AWS Deployment Guide

## Option 1: Elastic Beanstalk (Recommended - Easiest)

Elastic Beanstalk är managed service från AWS - du behöver bara pusha, resten hanterar AWS automatiskt.

### Prerequisites
- AWS account
- AWS CLI installed: https://aws.amazon.com/cli/
- Docker installed locally (for testing)

### Steps

1. **Installera EB CLI:**
```bash
pip install awsebcli
```

2. **Logga in på AWS:**
```bash
aws configure
# Ange: Access Key ID, Secret Access Key, Region (t.ex. eu-west-1)
```

3. **Initiera Elastic Beanstalk:**
```bash
eb init -p "Python 3.11 running on 64bit Amazon Linux 2" ml-assignment-2
# Välj region där du vill deplaya
```

4. **Skapa environment och deploya:**
```bash
eb create ml-assignment-2-env
# Vänta 5-10 minuter medan AWS sätter upp
```

5. **Visa URL:**
```bash
eb open
# Eller:
eb status
```

6. **Se logs:**
```bash
eb logs
```

7. **Uppdatera efter ändringar:**
```bash
git add .
git commit -m "Updates"
eb deploy
```

**Kostnad:** ~$10-15/månad för t3.micro (nära free tier)

---

## Option 2: EC2 + Docker (More Control)

### Prerequisites
- AWS account
- EC2 key pair created

### Steps

1. **Skapa EC2 instance:**
   - Go to AWS Console → EC2 → Launch Instance
   - Choose: Ubuntu 22.04 LTS (free tier eligible)
   - Instance type: t3.micro (free)
   - Create/select key pair
   - Security group: Allow HTTP (80), HTTPS (443), SSH (22)
   - Launch!

2. **SSH in to instance:**
```bash
ssh -i your-key.pem ubuntu@your-instance-ip
```

3. **Installera Docker:**
```bash
sudo apt-get update
sudo apt-get install -y docker.io docker-compose

# Lägg till ubuntu-user i docker-group
sudo usermod -aG docker ubuntu
exit
# SSH in igen för att aktivera
```

4. **Klona repo:**
```bash
git clone https://github.com/YOUR_USERNAME/ml-assignment-2.git
cd ml-assignment-2
```

5. **Kör med Docker:**
```bash
docker build -t ml-app .
docker run -d -p 80:8000 ml-app
```

6. **Testa:**
```bash
curl http://your-instance-ip/
```

7. **Setup domän (optional):**
   - Buy domain (Route 53, GoDaddy, etc.)
   - Point to EC2 Elastic IP
   - Setup SSL certificate (AWS ACM free)

**Kostnad:** ~$5-10/månd för t3.micro

---

## Option 3: AWS Lambda + API Gateway (Serverless)

Mer komplext för FastAPI, rekommenderas inte för denna app.

---

## Testing Locally with Docker

Innan du deplayer, testa lokalt:

```bash
# Build image
docker build -t ml-app .

# Run container
docker run -p 8000:8000 ml-app

# Testa
curl http://localhost:8000/
```

---

## Environment Variables (if needed in future)

Create `.ebextensions/python.config` for Elastic Beanstalk:

```yaml
option_settings:
  aws:elasticbeanstalk:container:python:
    WSGIPath: app.main:app
```

---

## Troubleshooting

### "Port 8000 already in use"
```bash
docker ps
docker stop container-id
```

### "Permission denied" errors
```bash
sudo usermod -aG docker $USER
# Log out and back in
```

### App crashes on deployment
Check logs:
```bash
eb logs
# or EC2:
docker logs container-id
```

---

## Recommendations

- **Start with:** Elastic Beanstalk (easiest)
- **If you want full control:** EC2 + Docker
- **Production:** Add load balancer, RDS database, etc.

Good luck! 🚀
