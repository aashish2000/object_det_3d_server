import numpy as np

def projection_mat(file_path):
	with open(file_path,"r") as f:
		content = f.readlines()

	extrinsic = np.transpose(np.reshape(np.fromstring(content[1].strip()[2:-2], sep=", ").astype(np.float32),(-1,4)))
	
	extrinsic *= np.array([		[ 1.0, -1.0, -1.0,  1.0],
                        		[-1.0,  1.0,  1.0, -1.0],
                               	[-1.0,  1.0,  1.0, -1.0],
                               	[ 1.0,  1.0,  1.0,  1.0]], dtype = np.float32)

	final_extrinsic = extrinsic[0:3][0:4]

	intrinsic_mama = np.array([[4.76464003e+03, 1.49280856e+01, 5.16860724e+02],
 [0.00000000e+00, 4.30386310e+03, 3.04258145e+03],
 [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])


	# intrinsic = np.array([[ 1.30609801e+03, -1.70266591e+00,  5.11800168e+02], [ 0.00000000e+00,  1.31574036e+03,  1.10658204e+03], [ 0.00000000e+00,  0.00000000e+00, 1.00000000e+00]])
	intrinsic_sourav = np.array([[ 3.97876660e+03, -4.87895214e+01,  2.54230438e+01],
 [ 0.00000000e+00,  3.97931790e+03,  2.82905708e+02],
 [ 0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
	# intrinsic = np.array([[4.28627017e+03, 1.47725457e+02, 1.80509276e+02],
	#  [0.00000000e+00, 4.00600618e+03, 2.57364938e+03],
	#  [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]], dtype=np.float32)
	intrinsic = intrinsic_sourav
	# intrinsic = np.array([[5.74906597e+03, 2.72691688e+01, 5.30390810e+02],
 # [0.00000000e+00, 4.88537018e+03, 3.97510802e+03],
 # [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

	# intrinsic = np.array([[1.01957428e+03, 0.00000000e+00, 5.82486145e+02],
 # [0.00000000e+00, 1.03348171e+03, 1.08826508e+03],
 # [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
	projection_matrix = np.matmul(intrinsic,final_extrinsic)
	# print(projection_matrix)
	return(projection_matrix)

def pose_to_projection(file_path):
	with open(file_path,"r") as f:
		content = f.readlines()

	out_pose = np.transpose(np.reshape(np.fromstring(content[2].strip()[2:-2], sep=", ").astype(np.float32),(-1,4)))
	rotation_transpose = np.delete(np.delete(np.transpose(out_pose), 3, 1), 3, 0)

	# rotation_transpose[0][3] = 0
	# rotation_transpose[1][3] = 0
	# rotation_transpose[2][3] = 0

	# print(rotation_transpose)

	intrinsic_mama = np.array([[4.76464003e+03, 1.49280856e+01, 5.16860724e+02],
 [0.00000000e+00, 4.30386310e+03, 3.04258145e+03],
 [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])


	# intrinsic = np.array([[ 1.30609801e+03, -1.70266591e+00,  5.11800168e+02], [ 0.00000000e+00,  1.31574036e+03,  1.10658204e+03], [ 0.00000000e+00,  0.00000000e+00, 1.00000000e+00]])
	intrinsic_sourav = np.array([[ 3.97876660e+03, -4.87895214e+01,  2.54230438e+01],
 [ 0.00000000e+00,  3.97931790e+03,  2.82905708e+02],
 [ 0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
	column = np.transpose([out_pose[:,3]*-1])[:3]

	# identity = np.delete(np.identity(4),3,1)
	# print("Column: ",column)

	rotation_column = np.matmul(rotation_transpose,column)
	extrinsic_matrix = np.concatenate((rotation_transpose,rotation_column), axis = 1)

	# print("Identity: ",  identity)

	# column_identity = np.concatenate((identity,column),axis = 1)
	# column_identity[3][3] *= -1

	# print(column_identity)
	# extrinsic_matrix = np.matmul(rotation_transpose,column_identity)
	# print(extrinsic_matrix)

	# intrinsic_matrix = np.array([[ 1.30609801e+03, -1.70266591e+00,  5.11800168e+02], [ 0.00000000e+00,  1.31574036e+03,  1.10658204e+03], [ 0.00000000e+00,  0.00000000e+00, 1.00000000e+00]])
	intrinsic_matrix = intrinsic_sourav

	projection_matrix = np.matmul(intrinsic_matrix,extrinsic_matrix)
	return(projection_matrix)
