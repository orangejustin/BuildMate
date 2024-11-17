# BuildMate
AI Construction Materials Advisor

A chatbot interface built with Python backend and React frontend.

## Prerequisites

- Python 3.10+
- Node.js 16+ (to use npm/pnpm)

## Installation

### 1. Backend Setup

```bash
# Create and activate Python environment
conda create -n mychatbot python=3.10
conda activate mychatbot

# Install Python dependencies
pip install -r requirements.txt
```

### 2. Frontend Setup

1. Install pnpm globally first:
```bash
npm install -g pnpm
```

2. Create a new React project with Vite:
```bash
pnpm create vite frontend --template react
cd frontend
```

3. Install base dependencies:
```bash
pnpm install
```

4. Install chatbot-specific dependencies:
```bash
pnpm install @ant-design/pro-chat antd antd-style --save
```

## Project Structure
```
mychatbot/
├── backend/
│   ├── requirements.txt
│   └── app.py
├── frontend/          # Created by Vite
│   ├── src/
│   │   ├── App.jsx
│   │   ├── main.jsx
│   │   └── components/
│   │       └── ChatbotInterface.jsx
│   ├── package.json
│   └── vite.config.js
└── README.md
```

## Running the Application

1. Start the backend server:
```bash
# From the backend directory
python app.py
```

2. Start the frontend development server:
```bash
# From the frontend directory
pnpm dev
```

The application will be available at `http://localhost:5173` (Vite's default port)