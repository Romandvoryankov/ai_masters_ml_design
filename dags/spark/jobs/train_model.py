from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS

spark = SparkSession.builder.appName("TrainModel").getOrCreate()
# hadoop_conf = spark.sparkContext._jsc.hadoopConfiguration()
# hadoop_conf.set("fs.s3a.connection.ssl.enabled", "true")
# hadoop_conf.set("fs.s3a.path.style.access", "true")
# hadoop_conf.set("fs.s3a.attempts.maximum", "1")
# hadoop_conf.set("fs.s3a.connection.establish.timeout", "5000")
# hadoop_conf.set("fs.s3a.connection.timeout", "10000")

# Загрузка данных
df = spark.read.csv("s3a://movielens/train.csv", header=True, inferSchema=True)

# Обучение модели
als = ALS(userCol="userId", itemCol="movieId", ratingCol="rating", coldStartStrategy="drop")
model = als.fit(df)

# Сохранение модели
model.write().overwrite().save("s3a://movielens/model")

spark.stop()