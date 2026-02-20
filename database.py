"""
Database module for user authentication
Uses SQLite for now, will migrate to MongoDB Atlas later
"""

import sqlite3
import hashlib
import secrets
from datetime import datetime
import os

DATABASE_PATH = 'health_system.db'

def get_db_connection():
    """Create database connection"""
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initialize database with users table"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Create users table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            salt TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_login TIMESTAMP,
            is_active INTEGER DEFAULT 1
        )
    ''')
    
    # Create user_profiles table for health data
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_profiles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            full_name TEXT,
            age INTEGER,
            gender TEXT,
            phone TEXT,
            last_assessment_date TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    # Create assessment_history table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS assessment_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            assessment_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            diabetes_risk REAL,
            heart_risk REAL,
            hypertension_risk REAL,
            obesity_risk REAL,
            health_score REAL,
            composite_risk REAL,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    conn.commit()
    conn.close()
    print("✅ Database initialized successfully")

def hash_password(password, salt=None):
    """Hash password with salt using SHA-256"""
    if salt is None:
        salt = secrets.token_hex(32)
    
    # Combine password and salt, then hash
    pwd_salt = password + salt
    pwd_hash = hashlib.sha256(pwd_salt.encode()).hexdigest()
    
    return pwd_hash, salt

def create_user(username, email, password):
    """Create new user account"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Check if username already exists
        cursor.execute('SELECT id FROM users WHERE username = ?', (username,))
        if cursor.fetchone():
            return {'success': False, 'error': 'Username already exists'}
        
        # Check if email already exists
        cursor.execute('SELECT id FROM users WHERE email = ?', (email,))
        if cursor.fetchone():
            return {'success': False, 'error': 'Email already registered'}
        
        # Hash password
        pwd_hash, salt = hash_password(password)
        
        # Insert user
        cursor.execute('''
            INSERT INTO users (username, email, password_hash, salt)
            VALUES (?, ?, ?, ?)
        ''', (username, email, pwd_hash, salt))
        
        user_id = cursor.lastrowid
        
        # Create empty profile
        cursor.execute('''
            INSERT INTO user_profiles (user_id)
            VALUES (?)
        ''', (user_id,))
        
        conn.commit()
        conn.close()
        
        return {'success': True, 'user_id': user_id, 'username': username}
    
    except Exception as e:
        return {'success': False, 'error': str(e)}

def verify_user(username, password):
    """Verify user credentials"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Get user data
        cursor.execute('''
            SELECT id, username, email, password_hash, salt, is_active
            FROM users
            WHERE username = ?
        ''', (username,))
        
        user = cursor.fetchone()
        
        if not user:
            return {'success': False, 'error': 'Invalid username or password'}
        
        if not user['is_active']:
            return {'success': False, 'error': 'Account is deactivated'}
        
        # Verify password
        pwd_hash, _ = hash_password(password, user['salt'])
        
        if pwd_hash != user['password_hash']:
            return {'success': False, 'error': 'Invalid username or password'}
        
        # Update last login
        cursor.execute('''
            UPDATE users
            SET last_login = CURRENT_TIMESTAMP
            WHERE id = ?
        ''', (user['id'],))
        
        conn.commit()
        conn.close()
        
        return {
            'success': True,
            'user_id': user['id'],
            'username': user['username'],
            'email': user['email']
        }
    
    except Exception as e:
        return {'success': False, 'error': str(e)}

def get_user_profile(user_id):
    """Get user profile data"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT u.username, u.email, u.created_at, u.last_login,
                   p.full_name, p.age, p.gender, p.phone, p.last_assessment_date
            FROM users u
            LEFT JOIN user_profiles p ON u.id = p.user_id
            WHERE u.id = ?
        ''', (user_id,))
        
        profile = cursor.fetchone()
        conn.close()
        
        if profile:
            return dict(profile)
        return None
    
    except Exception as e:
        print(f"Error getting profile: {e}")
        return None

def update_user_profile(user_id, full_name=None, age=None, gender=None, phone=None):
    """Update user profile information"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        updates = []
        values = []
        
        if full_name:
            updates.append('full_name = ?')
            values.append(full_name)
        if age:
            updates.append('age = ?')
            values.append(age)
        if gender:
            updates.append('gender = ?')
            values.append(gender)
        if phone:
            updates.append('phone = ?')
            values.append(phone)
        
        if updates:
            values.append(user_id)
            query = f"UPDATE user_profiles SET {', '.join(updates)} WHERE user_id = ?"
            cursor.execute(query, values)
            conn.commit()
        
        conn.close()
        return {'success': True}
    
    except Exception as e:
        return {'success': False, 'error': str(e)}

def save_assessment(user_id, diabetes_risk, heart_risk, hypertension_risk, obesity_risk, health_score, composite_risk):
    """Save assessment results to history"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO assessment_history 
            (user_id, diabetes_risk, heart_risk, hypertension_risk, obesity_risk, health_score, composite_risk)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (user_id, diabetes_risk, heart_risk, hypertension_risk, obesity_risk, health_score, composite_risk))
        
        # Update last assessment date in profile
        cursor.execute('''
            UPDATE user_profiles
            SET last_assessment_date = CURRENT_TIMESTAMP
            WHERE user_id = ?
        ''', (user_id,))
        
        conn.commit()
        conn.close()
        
        return {'success': True}
    
    except Exception as e:
        return {'success': False, 'error': str(e)}

def get_assessment_history(user_id, limit=10):
    """Get user's assessment history"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT assessment_date, diabetes_risk, heart_risk, hypertension_risk, 
                   obesity_risk, health_score, composite_risk
            FROM assessment_history
            WHERE user_id = ?
            ORDER BY assessment_date DESC
            LIMIT ?
        ''', (user_id, limit))
        
        history = [dict(row) for row in cursor.fetchall()]
        conn.close()
        
        return history
    
    except Exception as e:
        print(f"Error getting history: {e}")
        return []

# Initialize database when module is imported
if __name__ == '__main__':
    init_db()
