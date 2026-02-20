"""
Database module for user authentication using MongoDB Atlas
Maintains same function signatures as SQLite version for compatibility
"""

from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError
import hashlib
import secrets
from datetime import datetime
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# MongoDB connection
MONGODB_URI = os.getenv('MONGODB_URI')
DATABASE_NAME = 'health_system'

def get_db_connection():
    """Create MongoDB connection"""
    client = MongoClient(MONGODB_URI)
    return client[DATABASE_NAME]

def init_db():
    """Initialize MongoDB collections and indexes"""
    try:
        db = get_db_connection()
        
        # Create unique indexes
        db.users.create_index('username', unique=True)
        db.users.create_index('email', unique=True)
        db.assessments.create_index([('user_id', 1), ('assessment_date', -1)])
        
        print("✅ MongoDB database initialized successfully")
        print(f"📦 Database: {DATABASE_NAME}")
        print(f"📊 Collections: users, assessments")
        
    except Exception as e:
        print(f"❌ Error initializing MongoDB: {e}")

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
        db = get_db_connection()
        
        # Check if username already exists
        if db.users.find_one({'username': username}):
            return {'success': False, 'error': 'Username already exists'}
        
        # Check if email already exists
        if db.users.find_one({'email': email}):
            return {'success': False, 'error': 'Email already registered'}
        
        # Hash password
        pwd_hash, salt = hash_password(password)
        
        # Create user document
        user_doc = {
            'username': username,
            'email': email,
            'password_hash': pwd_hash,
            'salt': salt,
            'created_at': datetime.utcnow(),
            'last_login': None,
            'is_active': True,
            'profile': {
                'full_name': None,
                'age': None,
                'gender': None,
                'phone': None,
                'last_assessment_date': None
            }
        }
        
        # Insert user
        result = db.users.insert_one(user_doc)
        user_id = str(result.inserted_id)
        
        return {'success': True, 'user_id': user_id, 'username': username}
    
    except DuplicateKeyError:
        return {'success': False, 'error': 'Username or email already exists'}
    except Exception as e:
        return {'success': False, 'error': str(e)}

def verify_user(username, password):
    """Verify user credentials"""
    try:
        db = get_db_connection()
        
        # Get user data
        user = db.users.find_one({'username': username})
        
        if not user:
            return {'success': False, 'error': 'Invalid username or password'}
        
        if not user.get('is_active', True):
            return {'success': False, 'error': 'Account is deactivated'}
        
        # Verify password
        pwd_hash, _ = hash_password(password, user['salt'])
        
        if pwd_hash != user['password_hash']:
            return {'success': False, 'error': 'Invalid username or password'}
        
        # Update last login
        db.users.update_one(
            {'_id': user['_id']},
            {'$set': {'last_login': datetime.utcnow()}}
        )
        
        return {
            'success': True,
            'user_id': str(user['_id']),
            'username': user['username'],
            'email': user['email']
        }
    
    except Exception as e:
        return {'success': False, 'error': str(e)}

def get_user_profile(user_id):
    """Get user profile data"""
    try:
        from bson.objectid import ObjectId
        db = get_db_connection()
        
        user = db.users.find_one({'_id': ObjectId(user_id)})
        
        if not user:
            return None
        
        profile = user.get('profile', {})
        
        # Format to match SQLite structure for template compatibility
        return {
            'username': user['username'],
            'email': user['email'],
            'created_at': user['created_at'].strftime('%Y-%m-%d %H:%M:%S') if user.get('created_at') else None,
            'last_login': user['last_login'].strftime('%Y-%m-%d %H:%M:%S') if user.get('last_login') else None,
            'full_name': profile.get('full_name'),
            'age': profile.get('age'),
            'gender': profile.get('gender'),
            'phone': profile.get('phone'),
            'last_assessment_date': profile.get('last_assessment_date')
        }
    
    except Exception as e:
        print(f"Error getting profile: {e}")
        return None

def update_user_profile(user_id, full_name=None, age=None, gender=None, phone=None):
    """Update user profile information"""
    try:
        from bson.objectid import ObjectId
        db = get_db_connection()
        
        updates = {}
        
        if full_name is not None:
            updates['profile.full_name'] = full_name
        if age is not None:
            updates['profile.age'] = age
        if gender is not None:
            updates['profile.gender'] = gender
        if phone is not None:
            updates['profile.phone'] = phone
        
        if updates:
            db.users.update_one(
                {'_id': ObjectId(user_id)},
                {'$set': updates}
            )
        
        return {'success': True}
    
    except Exception as e:
        return {'success': False, 'error': str(e)}

def save_assessment(user_id, diabetes_risk, heart_risk, hypertension_risk, obesity_risk, health_score, composite_risk):
    """Save assessment results to history"""
    try:
        from bson.objectid import ObjectId
        db = get_db_connection()
        
        # Create assessment document
        assessment_doc = {
            'user_id': user_id,  # Store as string for easier querying
            'assessment_date': datetime.utcnow(),
            'diabetes_risk': float(diabetes_risk),
            'heart_risk': float(heart_risk),
            'hypertension_risk': float(hypertension_risk),
            'obesity_risk': float(obesity_risk),
            'health_score': float(health_score),
            'composite_risk': float(composite_risk)
        }
        
        # Insert assessment
        db.assessments.insert_one(assessment_doc)
        
        # Update last assessment date in user profile
        db.users.update_one(
            {'_id': ObjectId(user_id)},
            {'$set': {'profile.last_assessment_date': datetime.utcnow()}}
        )
        
        return {'success': True}
    
    except Exception as e:
        print(f"Error saving assessment: {e}")
        return {'success': False, 'error': str(e)}

def get_assessment_history(user_id, limit=10):
    """Get user's assessment history"""
    try:
        db = get_db_connection()
        
        # Query assessments
        assessments = db.assessments.find(
            {'user_id': user_id}
        ).sort('assessment_date', -1).limit(limit)
        
        # Format to match SQLite structure for template compatibility
        history = []
        for assessment in assessments:
            formatted_date = assessment['assessment_date'].strftime('%Y-%m-%d %H:%M:%S')
            history.append({
                'assessment_date': formatted_date,
                'created_at': formatted_date,  # Add for JS chart compatibility
                'diabetes_risk': assessment['diabetes_risk'],
                'heart_risk': assessment['heart_risk'],
                'hypertension_risk': assessment['hypertension_risk'],
                'obesity_risk': assessment['obesity_risk'],
                'health_score': assessment['health_score'],
                'composite_risk': assessment['composite_risk']
            })
        
        return history
    
    except Exception as e:
        print(f"Error getting history: {e}")
        return []

def test_connection():
    """Test MongoDB connection"""
    try:
        db = get_db_connection()
        # Ping the database
        db.command('ping')
        print("✅ MongoDB connection successful!")
        print(f"📍 Connected to: {DATABASE_NAME}")
        
        # Get collection stats
        users_count = db.users.count_documents({})
        assessments_count = db.assessments.count_documents({})
        
        print(f"👥 Users: {users_count}")
        print(f"📊 Assessments: {assessments_count}")
        
        return True
    except Exception as e:
        print(f"❌ MongoDB connection failed: {e}")
        return False

# Initialize database when module is imported
if __name__ == '__main__':
    print("🔧 Testing MongoDB connection...")
    if test_connection():
        print("\n🚀 Initializing database...")
        init_db()
        print("\n✅ Setup complete! Ready to use MongoDB Atlas.")
    else:
        print("\n❌ Please check your MONGODB_URI in .env file")
