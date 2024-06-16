CREATE TABLE CUSTOMER(
CustomerID INT PRIMARY KEY,
CustomerName VARCHAR(50) NOT NULL
);

CREATE TABLE Orders(
OrderId INT PRIMARY KEY,
OrderNumber INT NOT NULL,
CustomerID INT,
FOREIGN KEY (CustomerID) REFERENCES Customer(CustomerID)
);

INSERT INTO CUSTOMER VALUES
(1,'John'),
(2,'Jane'),
(3,'Bob');

INSERT INTO Orders VALUES
(1,101,1),
(2,202,2),
(3,103,3);

INSERT INTO Orders VALUES
(4,104,4);

CREATE TABLE EMP1(
EMP1ID INT PRIMARY KEY,
EMP1NAME VARCHAR(20),
EMP1COUNTRY VARCHAR(20)
);

CREATE TABLE EMP2(
EMP2ID INT PRIMARY KEY,
EMP2NAME VARCHAR(20),
EMP2COUNTRY VARCHAR(20)
);

INSERT INTO EMP1 VALUES
(1,'Shubham','India'),
(2,'Aman','Australia'),
(3,'Navam','Shri Lanka'),
(4,'Aditya','Austria'),
(5,'Nishant','Spain');

INSERT INTO EMP2 VALUES
(1,'Tommy','England'),
(2,'Allen','France'),
(3,'Nancy','India'),
(4,'Adi','Ireland'),
(5,'Sandy','Spain');

SELECT EMP1COUNTRY FROM EMP1
UNION
SELECT EMP2COUNTRY FROM EMP2
ORDER BY EMP1COUNTRY;

SELECT EMP1COUNTRY FROM EMP1
UNION ALL
SELECT EMP2COUNTRY FROM EMP2
ORDER BY EMP1COUNTRY;

SELECT EMP1COUNTRY FROM EMP1
INTERSECT 
SELECT EMP2COUNTRY FROM EMP2
ORDER BY EMP1COUNTRY;

SELECT EMP1COUNTRY FROM EMP1
EXCEPT
SELECT EMP2COUNTRY FROM EMP2
ORDER BY EMP1COUNTRY;

/* Inner join 
    SELECT COLUMN_NAME
    FROM TABLEA
    INNER JOIN TABLEB
    ON TABLEA.COLNAME=TABLEB.COLNAME
*/


CREATE TABLE CustomerJoin1 (
    customerid INT ,
    fname VARCHAR(50),
    lname VARCHAR(50),
    addressid INT
);

CREATE TABLE CustomerJoin2 (
    customerid INT,
    amount DECIMAL(10, 2),
    mode VARCHAR(20),
    date DATE
);


INSERT INTO CustomerJoin1 (customerid, fname, lname, addressid) VALUES 
(1, 'John', 'Doe', 101),
(2, 'Jane', 'Smith', 102),
(3, 'Alice', 'Johnson', 103),
(4, 'Bob', 'Brown', 104);


INSERT INTO CustomerJoin2 (customerid, amount, mode, date) VALUES 
(1, 100.00, 'Cash', '2023-01-01'),
(2, 150.00, 'Credit', '2023-01-02'),
(8, 200.00, 'Debit', '2023-01-03'),
(10, 250.00, 'Cash', '2023-01-04'),
(11, 300.00, 'Credit', '2023-01-05');

SELECT * 
FROM CustomerJoin1 AS c1
INNER JOIN CustomerJoin2 AS c2
ON c1.customerid = c2.customerid;

SELECT * 
FROM CustomerJoin1 AS c1
LEFT JOIN CustomerJoin2 AS c2
ON c1.customerid = c2.customerid;

SELECT * 
FROM CustomerJoin1 AS c1
RIGHT JOIN CustomerJoin2 AS c2
ON c1.customerid = c2.customerid;

/*
SELECT * 
FROM CustomerJoin1 AS c1
FULL OUTER JOIN CustomerJoin2 AS c2
ON c1.customerid = c2.customerid;
*/

SELECT *
FROM CustomerJoin1 AS c1
LEFT JOIN CustomerJoin2 AS c2 ON c1.customerid = c2.customerid
UNION
SELECT *
FROM CustomerJoin1 AS c1
RIGHT JOIN CustomerJoin2 AS c2 ON c1.customerid = c2.customerid;

/*
WINDOW FUNCTION
  SELECT columnname,
    fun() OVER([PARTITION BY CLAUSE]
			   [ORDER BY CLAUSE]
               [ROW OR RANGE CLAUSE])
  FROM Tablename
*/


CREATE TABLE Windows(
new_id INT ,
new_cat VARCHAR(20)
);

INSERT INTO Windows VALUES
(100,'Agni'),
(200,'Agni'),
(500,'Dharti'),
(700,'Dharti'),
(200,'Vayu'),
(300,'Vayu'),
(500,'Vayu');

SELECT *,
SUM(new_id) OVER (PARTITION BY new_cat ORDER BY new_id) AS "Total",
AVG(new_id) OVER (PARTITION BY new_cat ORDER BY new_id) AS "Average",
COUNT(new_id) OVER (PARTITION BY new_cat ORDER BY new_id) AS "Count",
MIN(new_id) OVER (PARTITION BY new_cat ORDER BY new_id) AS "Min",
MAX(new_id) OVER (PARTITION BY new_cat ORDER BY new_id) AS "Max"
FROM Windows;

SELECT *,
SUM(new_id) OVER (ORDER BY new_id ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS "Total",
AVG(new_id) OVER (ORDER BY new_id ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS "Average",
COUNT(new_id) OVER (ORDER BY new_id ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS "Count",
MIN(new_id) OVER (ORDER BY new_id ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS "Min",
MAX(new_id) OVER (ORDER BY new_id ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS "Max"
FROM Windows



