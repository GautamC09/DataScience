Select @@version;
CREATE DATABASE DATASCIENCE;
USE DATASCIENCE;

CREATE TABLE EMP(
  EMPNAME VARCHAR(100),
  EMPCITY VARCHAR(100),
  EMPID INT PRIMARY KEY,
  EMPCONTACT INT,
  EMPSALARY INT
);

INSERT INTO EMP (EMPNAME, EMPCITY, EMPID, EMPCONTACT, EMPSALARY) VALUES
('John Doe', 'New York', 1, 1234567890, 60000),
('Jane Smith', 'Los Angeles', 2, 987654321, 75000),
('Alice Johnson', 'Chicago', 3, 555123456, 65000),
('Bob Brown', 'Houston', 4, 444987654, 80000),
('Charlie Davis', 'Phoenix', 5, 333555987, 70000);

INSERT INTO DS_EMP (EMPNAME, EMPCITY, EMPID, EMPCONTACT, EMPSALARY) VALUES
('John Foe', 'New York', 6, 1234567880, 60000),
('Mob Brown', 'Houston', 7, 444987655, 80000);

ALTER TABLE EMP
ADD COLUMN EMPAGE INT;

RENAME TABLE EMP TO DS_EMP;

SELECT * FROM DS_EMP;

SELECT * FROM DS_EMP
WHERE EMPNAME='BOB BROWN';

CREATE TABLE TEMP_EMP AS
SELECT * FROM DS_EMP;

CREATE TEMPORARY TABLE TEMPTABLE (
    TEMPID INT,
    TEMPNAME VARCHAR(100)
);

INSERT INTO TEMPTABLE (TEMPID, TEMPNAME)
VALUES
    (1, 'Alpha'),
    (2, 'Bravo'),
    (3, 'Charlie'),
    (4, 'Delta'),
    (5, 'Echo');

SELECT * FROM TEMPTABLE;

SELECT COUNT(*) AS TotalCount FROM DS_EMP;

SELECT SUM(EMPSALARY) AS TotalSalary FROM DS_EMP;

SELECT AVG(EMPSALARY) AS AverageSalary FROM DS_EMP;

SELECT MIN(EMPSALARY) AS MinimumSalary FROM DS_EMP;

SELECT MAX(EMPSALARY) AS MaximumSalary FROM DS_EMP;

SELECT * FROM DS_EMP
WHERE EMPSALARY BETWEEN 60000 AND 70000;

SELECT EMPNAME, EMPCITY,EMPID FROM DS_EMP
WHERE EMPNAME LIKE 'J%';

SELECT * FROM DS_EMP
WHERE EMPCITY LIKE 'N%' OR EMPCITY LIKE 'L%';

SELECT * FROM DS_EMP
WHERE EMPCITY  IN ('New York', 'Los Angeles');

SELECT * FROM DS_EMP
WHERE EMPCITY='New York' AND EMPSALARY>50000;

SELECT * FROM DS_EMP
WHERE EMPNAME NOT LIKE 'J%';

SELECT * FROM DS_EMP
WHERE EMPCITY <> 'New York';

SELECT EMPNAME,EMPSALARY FROM DS_EMP
ORDER BY EMPSALARY DESC;

SELECT EMPCITY,
       COUNT(*) AS TotalEmployees,
       SUM(EMPSALARY) AS TotalSalary
FROM DS_EMP
GROUP BY EMPCITY;

CREATE TABLE PAYMENT(
PID INT,
PAMT INT,
PMODE VARCHAR(20),
PDATE DATE
);


INSERT INTO PAYMENT (PID, PAMT, PMODE, PDATE)
VALUES
    (1, 500, 'Credit Card', '2024-06-01'),
    (2, 750, 'Cash', '2024-06-02'),
    (3, 600, 'Debit Card', '2024-06-03'),
    (4, 900, 'Cash', '2024-06-04'),
    (5, 800, 'Credit Card', '2024-06-05'),
    (6, 700, 'Debit Card', '2024-06-06');
    
SELECT PMODE, SUM(PAMT) AS TOTAL
FROM PAYMENT
GROUP BY PMODE
HAVING SUM(PAMT)>800;


CREATE TABLE Employee (
    EmployeeID INT PRIMARY KEY,
    Name VARCHAR(50),
    Gender CHAR(1),
    Department VARCHAR(50),
    Education VARCHAR(50),
    MonthOfJoining VARCHAR(20),
    Salary INT
);




INSERT INTO Employee (EmployeeID, Name, Gender, Department, Education, MonthOfJoining, Salary) VALUES
(1001, 'Ajay', 'M', 'Engineering', 'Doctoral', 'January', 25),
(1002, 'Babloo', 'M', 'Engineering', 'UG', 'February', 23),
(1003, 'Chhavli', 'F', 'HR', 'PG', 'March', 15),
(1004, 'Dheeraj', 'M', 'HR', 'UG', 'January', 12),
(1005, 'Evina', 'F', 'Marketing', 'UG', 'March', 16),
(1006, 'Fredy', 'M', 'Sales', 'UG', 'December', 10),
(1007, 'Garima', 'F', 'Sales', 'PG', 'March', 10),
(1008, 'Hans', 'M', 'Admin', 'PG', 'November', 8),
(1009, 'Ivanka', 'F', 'Admin', 'Intermediate', 'April', 7),
(1010, 'Jai', 'M', 'Peon', 'High School', 'December', 4);

SELECT Department,SUM(SALARY) AS TOTAL FROM Employee WHERE Education !='UG' 
GROUP BY Department
HAVING SUM(Salary) >= 20;

SELECT Department,SUM(SALARY) AS TOTAL FROM Employee 
GROUP BY Department
HAVING SUM(Salary) >= 20
ORDER BY TOTAL DESC;

SELECT NAME, Salary+5 FROM Employee;
