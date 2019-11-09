DROP TABLE IF EXISTS sales;
CREATE TEMPORARY TABLE sales
(
    key       varchar(5),
    ts        timestamp,
    product   integer,
    completed boolean,
    price     float
);

INSERT INTO sales
VALUES ('sale_1', '2019-11-08 00:00', 0, TRUE, 1.1),
       ('sale_2', '2019-11-08 01:00', 0, FALSE, 1.2),
       ('sale_3', '2019-11-08 01:00', 0, TRUE, 1.3),
       ('sale_4', '2019-11-08 01:00', 1, FALSE, 1.4),
       ('sale_5', '2019-11-08 02:00', 1, TRUE, 1.5),
       ('sale_6', '2019-11-08 02:00', 1, TRUE, 1.5);

-- Averages of binary columns
SELECT avg(product)
FROM sales;



-- Averages with decode
SELECT avg(price)
FROM (SELECT case when completed = TRUE then price else 0 end AS price FROM sales) AS q1;

SELECT avg(price)
FROM (SELECT case when completed = TRUE then price else NULL end AS price FROM sales) AS q1;

-- Timestamps
SELECT product, price
FROM (SELECT product, price, row_number() OVER (PARTITION BY product ORDER BY ts DESC) AS ix FROM sales) AS q1
WHERE ix = 1;

-- Order by
SELECT product, price
FROM (SELECT product, price, row_number() OVER (PARTITION BY product ORDER BY ts, key DESC) AS ix FROM sales) AS q1
WHERE ix = 1;


SELECT product, price
FROM (SELECT product, price, row_number() OVER (PARTITION BY product ORDER BY ts DESC, key) AS ix FROM sales) AS q1
WHERE ix = 1;



DROP TABLE IF EXISTS hourly_delay;
CREATE TEMPORARY TABLE hourly_delay
(
    ts    timestamp,
    delay float
);


INSERT INTO hourly_delay
VALUES ('2019-11-08 00:00', 80.1),
       ('2019-11-08 01:00', 100.2),
       ('2019-11-08 02:00', 70.3);


-- Joins
SELECT t2.ts::DATE, sum(t2.delay), avg(t1.price)
FROM hourly_delay AS t2
         INNER JOIN sales AS t1 ON t1.ts = t2.ts
GROUP BY t2.ts::DATE;


SELECT t1.ts, daily_delay, avg_price
FROM (SELECT t2.ts::DATE, sum(t2.delay) AS daily_delay FROM hourly_delay AS t2 GROUP BY t2.ts::DATE) AS t2
         INNER JOIN (SELECT ts::DATE AS ts, avg(price) AS avg_price FROM sales GROUP BY ts::DATE) AS t1 ON t1.ts = t2.ts;
