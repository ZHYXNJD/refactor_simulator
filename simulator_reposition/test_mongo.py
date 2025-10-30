import pymongo
from pymongo.errors import ConnectionFailure

myclient = pymongo.MongoClient("mongodb://host.docker.internal:27017/")
try:
    # The ismaster command is cheap and does not require auth.
    myclient.admin.command('ismaster')
    print("MongoDB is connected!")
except ConnectionFailure:
    print("Server not available")