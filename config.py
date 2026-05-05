import os


class Config:
    SECRET_KEY = "dev-secret-key-123"
    SQLALCHEMY_DATABASE_URI = "mysql+pymysql://root:@localhost/tccs_db"
    SQLALCHEMY_TRACK_MODIFICATIONS = False
