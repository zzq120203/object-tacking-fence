host = "localhost"
port = 1

# sentinel
db_urls = [("10.148.19.25", 46380),
           ("10.148.19.26", 46380),
           ("10.148.19.27", 46380)]
db_password = "wbqixgs"
# db_master_name == null ? redis : sentinel
db_master_name = "wbqmaster"