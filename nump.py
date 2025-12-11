import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

num_students=300

#generating random data
math_score=np.random.randint(0,101,num_students)
eng_score=np.random.randint(0,101,num_students)
science=np.random.randint(0,101,num_students)
age=np.random.randint(0,21,num_students)

#combining our genarated arrays into 2D Array
data=np.column_stack((math_score,eng_score,science,age))
data=data.astype(float)

#introducing ,issing values
missing_values=np.random.choice(data.size, 21, replace=False)
data_flat=data.flatten()
data_flat[missing_values]=np.nan
data=data_flat.reshape(300,4)

#Replacing missing values
col_mean=np.nanmean(data,axis=0)
inds=np.where(np.isnan(data))
data[inds]=np.take(col_mean,inds[1])

#clip scores between 0-100
data[:,0:3]=np.clip(data[:,0:3],0,100)

#compute analytics
sub_means=np.mean(data[:,0:3], axis=0)
sub_max=np.max(data[:,0:3], axis=0)
sub_min=np.min(data[:,0:3], axis=0)

avg_sore= np.mean(data[:,0:3],axis=1)
best_student=np.argmax(avg_sore)

passes= avg_sore >= 50

weights=np.array([0.4, 0.3, 0.3])

perf_scores=np.dot(data[:,0:3],weights)

z_scores=(data[:,0:3] - sub_means) / np.std(data[:,0:3], axis=0)
outliers= np.abs(z_scores) > 2.5

plt.hist(data[:,0], bins=20)
plt.title("Math Score Distribution")
plt.xlabel("Score")
plt.ylabel("Frequency")
plt.show()

np.save("students_clean.npy",data)
np.savetxt("students_clean.csv", data, delimiter=",",fmt="%.2f")


#print("Project Completed.")
#print("Subject means:", sub_means)
#print("Best student index:", best_student)
#print("Outlier counts:",np.sum(outliers, axis=0))
print(math_score.shape)