/*
  Copyright 2012 by Dr. Vlasios Voudouris and ABM Analytics Ltd
  Licensed under the Academic Free License version 3.0
  See the file "LICENSE" for more information
*/
package gamlss.utilities;

import java.io.BufferedWriter;
import java.io.FileWriter;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.BlockRealMatrix;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.QRDecomposition;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.util.FastMath;

/**
 * 01/08/2012
 * @author Dr. Vlasios Voudouris, Daniil Kiose, 
 * Prof. Mikis Stasinopoulos and Dr Robert Rigby.
 *
 */
public final class MatrixFunctions {
	
	public static ArrayRealVector multVvsD(ArrayRealVector v, double d) {
	     int size = v.getDimension();
		 double[] out = new double[size];
		 for (int i = 0; i < size; i++) {
			 out[i] = v.getEntry(i) * d;
		 }
		 return new ArrayRealVector(out, false);		 
	}
	
	public static ArrayRealVector addV(ArrayRealVector v1, ArrayRealVector v2) {
	     int size = v1.getDimension();
		 double[] out = new double[size];
		 for (int i = 0; i < size; i++) {
			 out[i] = v1.getEntry(i) + v2.getEntry(i);
		 }
		 return new ArrayRealVector(out, false);		 
	}
	
	public static double dotProduct(ArrayRealVector v1, ArrayRealVector v2) {
		 double out = 0;
		 for (int i = 0; i < v1.getDimension(); i++) {
			 out = out + v1.getEntry(i)*v2.getEntry(i);
		 }
		 return out;		 
	}

	/**
	 * if the length of the return matrix is zero,
	 *  then the diff was not completed properly.
	 *  @param data - matrix
	 *  @return - matrix
	 */
	private static BlockRealMatrix getDiff(final BlockRealMatrix data) {
		 double[][] newdata = new double[data.getRowDimension() - 1]
				 							[data.getColumnDimension()];
		 for (int i = 0; i < newdata.length; i++) {
			 for (int j = 0; j < newdata[i].length; j++) {
				 newdata[i][j] = data.getEntry(i + 1, j) - data.getEntry(i, j);
			 }
		 }
		 return new BlockRealMatrix(newdata);		 
	}
	


	/**
	 * Calculate this: (x - t) ^ p * (x > t).
	 * @param v1 vector 1
	 * @param v2 vector 2
	 * @param p - power
	 * @return  matrix C with Cij = v1i * v2j.
	 */
	//tpower <- function(x, t, p)
	public static BlockRealMatrix  outertpowerPB(final ArrayRealVector v1,
			  					      final ArrayRealVector v2,  final int p) {
		  
		  double[][] newdata = new double[v1.getDimension()]
				  									[v2.getDimension()];
		  for (int j = 0; j < v1.getDimension(); j++) {
				for (int j2 = 0; j2 < v2.getDimension(); j2++) {
					if (v1.getEntry(j) > v2.getEntry(j2)) {
						newdata[j][j2] = FastMath.pow((v1.getEntry(j) 
													- v2.getEntry(j2)), p);
					} else {
						newdata[j][j2] = 0d;
					}
				}
		  }
		  return new BlockRealMatrix(newdata);
	}
	
	/**
	 * Build identity matrix.
	 * @param size - size of the matrix
	 * @return identity matrix
	 */
		public static BlockRealMatrix buildIdentityMatrix(final int size) {
			double[] tempArr = new double[size]; 
			for (int i = 0; i < size; i++) {
				tempArr[i] = 1;
			}
			return new BlockRealMatrix(
					MatrixUtils.createRealDiagonalMatrix(tempArr).getData());
		}
		
	 /**
	 * if the length of the return matrix is zero, 
	 * then the function was not completed properly. 
	 * Check the 'diff' operator. 
	 * @param data - matrix
	 * @param diff - difference value
	 * @return matrix
	 */
	  public static BlockRealMatrix diff(final BlockRealMatrix data, 
			  											final int diff) {
		 BlockRealMatrix newdata 
		 			= new BlockRealMatrix(data.getRowDimension() - 1,
		 									 data.getColumnDimension());
		 for (int i = 0; i < diff; i++) {
			  if (i == 0) {
			     newdata = getDiff(data);
			  } else {
				  newdata = getDiff(newdata); 
			  }
		 }		 		 
		 return newdata;		 
	  }
	  

		 /**
		 * if the length of the return vector is zero, 
		 * then the function was not completed properly. 
		 * Check the 'diff' operator. 
		 * @param data - matrix
		 * @param diff - difference value
		 * @return matrix
		 */
		  public static ArrayRealVector diffV(final ArrayRealVector data, 
				  											final int diff) {
			  ArrayRealVector newdata 
			 			= new ArrayRealVector(data.getDimension() - 1);

			 for (int i = 0; i < diff; i++) {
				  if (i == 0) {
				     newdata = getDiff(data);
				  } else {
					  newdata = getDiff(newdata); 
				  }
			 }		 		 
			 return newdata;		 
		  }
		  
			/**
			 * if the length of the return vector is zero,
			 *  then the diff was not completed properly.
			 *  @param data - vector
			 *  @return - vector
			 */
			private static ArrayRealVector getDiff(final ArrayRealVector data) {
				 double[] newdata = new double[data.getDimension() - 1];

				 for (int i = 0; i < newdata.length; i++) {
						 newdata[i] = data.getEntry(i + 1) - data.getEntry(i);
					 }
				 return new ArrayRealVector(newdata, false);		 
			}
	  
	  /**
	   * Prints matrix values in the console.
	   * @param m - matrix to print
	   */
	  public static void matrixPrint(final BlockRealMatrix m) {
		    for (int i = 0; i < m.getRowDimension(); i++) {
		    	for (int j = 0; j < m.getColumnDimension(); j++) {
		    		System.out.print(m.getEntry(i, j));
		    		System.out.print(" ");
		    	}
		    	System.out.println();
		    }
		    System.out.println(" ");
	 }
	
	  /**
	   * Prints vector values in the console.
	   * @param v - vector to print
	   */
	 public static void vectorPrint(final ArrayRealVector v) {
		    for (int i = 0; i < v.getDimension(); i++) {
		    		System.out.println(v.getEntry(i));
		    }
	 }	
		
	 /**
	  * Write matrix values to CSV file.
	  * @param cmd - path to the file
	  * @param m - matrix to write
	  */
	 public static void matrixWriteCSV(final String cmd,
			 							    final BlockRealMatrix m) {
		 try {
				// Create file 
				FileWriter fstream = new FileWriter(cmd, false);
				BufferedWriter out = new BufferedWriter(fstream);
				
				for (int i = 0; i < m.getRowDimension(); i++) {
			    	for (int j = 0; j < m.getColumnDimension(); j++) {
			    		out.write(Double.toString(m.getEntry(i, j)));
			    		if (j < m.getColumnDimension() - 1) {
			    		out.append(',');
			    		}
			    	}
			    	out.newLine();
			    }
				out.close();
			} catch (Exception e) { //Catch exception if any
				System.err.println("Error: " + e.getMessage());
			}
	 }
	 
	 /**
	  * Write vector values to CSV file.
	  * @param cmd - path to the file
	  * @param v - vector to write
	  */
	public static void vectorWriteCSV(final String cmd, 
											final RealVector v, boolean append) {
		 try {
				// Create file 
				FileWriter fstream = new FileWriter(cmd, append);
				BufferedWriter out = new BufferedWriter(fstream);
				
			    	for (int j = 0; j < v.getDimension(); j++) {
			    		out.write(Double.toString(v.getEntry(j)));
			    		out.newLine();
			    	}
				out.close();
			} catch (Exception e) { //Catch exception if any
				System.err.println("Error: " + e.getMessage());
			}
	 }

	 /**
	  * Write value  to CSV file.
	  * @param cmd - path to the file
	  * @param d - value
	  */
	public static void doubleWriteCSV(final String cmd, 
													final double d,
													final boolean append) {
		 
		 try {
				// Create file 
				FileWriter fstream = new FileWriter(cmd, append);
				BufferedWriter out = new BufferedWriter(fstream);
				
			    out.write(Double.toString(d));
			    out.newLine();
				out.close();
			} catch (Exception e) { //Catch exception if any
				System.err.println("Error: " + e.getMessage());
			}
	}

	/**
	 * Append rows of the matrices.
	 * @param m1 - first matrix
	 * @param m2 - second matrix
	 * @return m1.append.m2
	 */
	public static BlockRealMatrix appendMatricesRows(final BlockRealMatrix m1,
													final BlockRealMatrix m2) {
		BlockRealMatrix out = new BlockRealMatrix(m1.getRowDimension(), 
						m1.getColumnDimension() + m2.getColumnDimension());
		for (int i = 0; i < m1.getRowDimension(); i++) {
	    	out.setRowVector(i, m1.getRowVector(i).append(m2.getRowVector(i))); 
	    } 
		return out;
	}
	
	/**
	 * Create a matrix with length of y number of rows
	 *  and one column, all entries of the matrix are 1.
	 * @param y - vector 
	 * @return matrix
	 */
	public static BlockRealMatrix setInterceptMatrix(final ArrayRealVector y) {
		 BlockRealMatrix designMatrix 
		 						= new BlockRealMatrix(y.getDimension(), 1);
		 for (int i = 0; i < y.getDimension(); i++) {
			 designMatrix.setEntry(i, 0, 1.0);
		 }	
		 return designMatrix;
	}
	
	 /**
	* <p>Compute the "hat" matrix.
	* </p>
	* <p>The hat matrix is defined in terms of the design matrix X
	*  by X(X<sup>T</sup>X)<sup>-1</sup>X<sup>T</sup>
	* </p>
	* <p>The implementation here uses the QR decomposition to compute the
	* hat matrix as Q I<sub>p</sub>Q<sup>T</sup> where I<sub>p</sub> is the
	* p-dimensional identity matrix augmented by 0's.  This computational
	* formula is from "The Hat Matrix in Regression and ANOVA",
	* David C. Hoaglin and Roy E. Welsch,
	* <i>The American Statistician</i>, Vol. 32, No. 1 (Feb., 1978), pp. 17-22.
	*@param m - matrix
	* @return the hat matrix
	*/
	public static RealMatrix calculateHat(final BlockRealMatrix m) {
		QRDecomposition qr = new QRDecomposition(m);
	   // Create augmented identity matrix
	   RealMatrix qM = qr.getQ();
	   final int p = qr.getR().getColumnDimension();
	   final int n = qM.getColumnDimension();
	   Array2DRowRealMatrix augI = new Array2DRowRealMatrix(n, n);
	   double[][] augIData = augI.getDataRef();
	   for (int i = 0; i < n; i++) {
	       for (int j = 0; j < n; j++) {
	           if (i == j && i < p) {
	               augIData[i][j] = 1d;
	           } else {
	               augIData[i][j] = 0d;
	           }
	       }
	   }
	   // Compute and return Hat matrix
	   return qM.multiply(augI).multiply(qM.transpose());
	}	
	
	/**
	*  Calculates inverse of vector values.
	* @param v - vector
	* @return 1/v
	*/
	public static ArrayRealVector inverse(final ArrayRealVector v) {
		double[] tempArr = new double[v.getDimension()];
			for (int i = 0; i < tempArr.length; i++) {
				tempArr[i] = 1 / v.getEntry(i);
			}	
			return new ArrayRealVector(tempArr, false);
	}
	
	/**
	 * Add value to each entry of vector.
	 * @param v - vector
	 * @param value - value to add
	 * @return new vector
	 */
	public static ArrayRealVector addValueToVector(final ArrayRealVector v,
														 final double value) {
		double[] tempArr = new double[v.getDimension()];
		for (int i = 0; i < tempArr.length; i++) {
			tempArr[i] = v.getEntry(i) + value;
		}
		return  new ArrayRealVector(tempArr, false);
	}
	
	/**
	* Calculates a difference of two vectors (v1 - v2).
	* @param v1 - first vector
	* @param v2 - second vector
	* @return vector (v1 - v2)
	*/
	public static ArrayRealVector vecSub(final ArrayRealVector v1, 
													 final ArrayRealVector v2) {
		int size = v1.getDimension();
		double[] tempArr = new double[size];
		for (int i = 0; i < size; i++) {   
			tempArr[i] = v1.getEntry(i) - v2.getEntry(i);
		}
		return new ArrayRealVector(tempArr, false);
	}
	
	/**
	* Calculates the sum of two vectors (v1+v2).
	* @param v1 - first vector
	* @param v2 - second vector
	* @return vector (v1+v2)
	*/
	public static ArrayRealVector vecSum(final ArrayRealVector v1, 
													final ArrayRealVector v2) {
		int size = v1.getDimension();
		double[] tempArr = new double[size];
		for (int i = 0; i < size; i++) {   
			tempArr[i] = v1.getEntry(i) + v2.getEntry(i);
		} 
		return new ArrayRealVector(tempArr, false); 
	}
	
	/**
	* Sum all vector entries.
	* @param v - vector
	* @return sum of vector entries
	*/
	public static double sumV(final ArrayRealVector v) {
			double tempD = 0;
			for (int j = 0; j < v.getDimension(); j++) {
				tempD = tempD + v.getEntry(j);
			}
		return tempD; 
	}
	
	/**
	* Create a vector of supplied length with supplied 
	* value at every vector entry.
	* @param value - value of every vector entry
	* @param lenght - length of vector
	* @return new vector 
	*/
	public static ArrayRealVector repV(final double value, final int lenght) {
		ArrayRealVector tempV = new ArrayRealVector(lenght);
		tempV.set(value);
		return tempV; 
	}
	
	/**
	 * Get main diagonal of the matrix.
	 * @param m - matrix
	 * @return main diagonal as vector
	 */
	public static ArrayRealVector getMainDiagonal(final BlockRealMatrix m) {
			double[] tempArr = new double[m.getRowDimension()];
		for (int i = 0; i < m.getColumnDimension(); i++) {
			for (int j = 0; j < m.getColumnDimension(); j++) {
				if (i == j) {
					tempArr[i] = m.getEntry(i, j);
				}
			}
		}
			return new ArrayRealVector(tempArr, false);
		}
	
	/**
	* Calculate sqrt of the the vector entries.
	* @param v - vector
	* @return sqrt(vector)
	*/
		public static ArrayRealVector sqrtVec(final ArrayRealVector v) {
		  int size    = v.getDimension();
		  double[] tempArr = new double[size];
		  for (int i = 0; i < size; i++) {
			tempArr[i] = FastMath.sqrt(v.getEntry(i));
		  }
		  return new ArrayRealVector(tempArr, false);
		}
		
		/**
		 * Multiply vector with every row of the matrix.
		 * @param v - vector
		 * @param m - matrix
		 * @return v*m
		 */
	public static BlockRealMatrix multVectorMatrix(final ArrayRealVector v,
													  final BlockRealMatrix m) {
			
		BlockRealMatrix tempM = new BlockRealMatrix(m.getRowDimension(),
													m.getColumnDimension());
			for (int i = 0; i < m.getColumnDimension(); i++) {
				tempM.setColumnVector(i, v.ebeMultiply(m.getColumnVector(i)));
			}
			return tempM;
		}	
	
    /**
     * Build intercept matrix of dimension dim
     * @param dim - number of rows of intercept matrix
     * @return intercept matrix 
     */
	public static BlockRealMatrix buildInterceptMatrix(int dim) {
		 BlockRealMatrix designMatrix = new BlockRealMatrix(dim, 1);
		 for (int i = 0; i < dim; i++) {
			 designMatrix.setEntry(i, 0, 1.0);
		 }	
		 return designMatrix;
	 } 
	
	/**
	* Calculate log of the the vector entries.
	* @param v - vector
	* @return log(vector)
	*/
		public static ArrayRealVector logVec(final ArrayRealVector v) {
		  int size    = v.getDimension();
		  double[] tempArr = new double[size];
		  for (int i = 0; i < size; i++) {
			tempArr[i] = FastMath.log(v.getEntry(i));
		  }
		  return new ArrayRealVector(tempArr, false);
		}
	
}
