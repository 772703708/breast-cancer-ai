import sqlite3
import hashlib
import os

def create_admin():
    os.makedirs('database', exist_ok=True)
    conn = sqlite3.connect('database/db.sqlite3')
    c = conn.cursor()
    
    # Check if admin already exists
    c.execute('SELECT * FROM users WHERE email = ?', ('admin@breastcancerai.com',))
    existing = c.fetchone()
    
    if existing:
        print("Admin account already exists!")
        print(f"Email: admin@breastcancerai.com")
        print("Use 'Forgot Password' to reset if needed")
    else:
        # Create admin account
        password = hashlib.sha256('admin123'.encode()).hexdigest()
        c.execute('''
            INSERT INTO users (fullname, username, email, password, created_at)
            VALUES (?, ?, ?, ?, datetime('now'))
        ''', ('System Administrator', 'admin', 'admin@breastcancerai.com', password))
        conn.commit()
        print("="*50)
        print("ADMIN ACCOUNT CREATED SUCCESSFULLY")
        print("="*50)
        print(f"Email: admin@breastcancerai.com")
        print(f"Password: admin123")
        print("="*50)
    
    conn.close()

if __name__ == '__main__':
    create_admin()