import os
import json
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from frontend.app import User, RememberedUser, Base

def migrate():
    frontend_dir = os.path.join(os.path.dirname(__file__), 'frontend')
    db_path = os.path.join(frontend_dir, 'users_data.db')
    users_json_path = os.path.join(frontend_dir, 'users_data.json')
    rem_json_path = os.path.join(frontend_dir, 'remember_me.json')
    
    DB_URL = f"sqlite:///{db_path}"
    engine = create_engine(DB_URL, echo=False)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    
    # Ensure tables exist
    Base.metadata.create_all(bind=engine)
    
    with SessionLocal() as db:
        # Migrate users
        if os.path.exists(users_json_path):
            with open(users_json_path, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                    for email, state in data.items():
                        user = db.query(User).filter(User.email == email.lower()).first()
                        if not user:
                            user = User(email=email.lower())
                            db.add(user)
                        user.state_json = json.dumps(state)
                    db.commit()
                    print(f"Migrated {len(data)} users from {users_json_path}")
                except Exception as e:
                    print(f"Error migrating users: {e}")
        else:
            print(f"No {users_json_path} found to migrate.")
            
        # Migrate remembered user
        if os.path.exists(rem_json_path):
            with open(rem_json_path, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                    email = data.get('email')
                    if email:
                        rem = db.query(RememberedUser).filter(RememberedUser.id == "current").first()
                        if not rem:
                            rem = RememberedUser(id="current")
                            db.add(rem)
                        rem.email = email
                        db.commit()
                        print(f"Migrated remembered user from {rem_json_path}")
                except Exception as e:
                    print(f"Error migrating remembered user: {e}")
        else:
            print(f"No {rem_json_path} found to migrate.")
            
    print("Migration complete. You can safely delete users_data.json and remember_me.json once you verify it works.")

if __name__ == '__main__':
    migrate()
