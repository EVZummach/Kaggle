import pyodbc

server = 'localhost,1433'
database = 'master'
username = 'sa'
password = 'DB_Titanic'
cnxn = pyodbc.connect('DRIVER={ODBC Driver 17 for SQL Server};SERVER='+server+';DATABASE='+database+';UID='+username+';PWD='+ password)
cursor = cnxn.cursor()

cursor = cnxn.cursor()

cnxn.commit()

cursor.execute('''
		INSERT INTO teste (product_id, product_name, price)
		VALUES
			(1,'Desktop Computer',800),
			(2,'Laptop',1200),
			(3,'Tablet',200),
			(4,'Monitor',350),
			(5,'Printer',150)
                ''')
cnxn.commit()