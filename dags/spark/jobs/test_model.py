from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALSModel

spark = SparkSession.builder.appName("TestModel").getOrCreate()

# Загрузка модели и данных
model = ALSModel.load("s3a://movielens/model")
test_df = spark.read.csv("s3a://movielens/test.csv", header=True, inferSchema=True)

# Генерация предсказаний
predictions = model.transform(test_df)

# Вычисление RMSE
from pyspark.sql.functions import col
predictions = predictions.withColumn("error", (col("rating") - col("prediction")) ** 2)
rmse = predictions.selectExpr("sqrt(avg(error)) as rmse").collect()[0]["rmse"]
print(f"RMSE: {rmse}")
