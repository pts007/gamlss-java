/*
  Copyright 2012 by Dr. Vlasios Voudouris and ABM Analytics Ltd
  Licensed under the Academic Free License version 3.0
  See the file "LICENSE" for more information
  
  Dr. Vlasios Voudouris, Daniil Kiose, Prof. Mikis Stasinopoulos and Dr Robert Rigby
*/
package gamlss.utilities;


import org.apache.commons.math3.analysis.function.Sqrt;
import org.apache.commons.math3.exception.DimensionMismatchException;
import org.apache.commons.math3.exception.NoDataException;
import org.apache.commons.math3.exception.NullArgumentException;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.BlockRealMatrix;
import org.apache.commons.math3.linear.LUDecomposition;
import org.apache.commons.math3.linear.QRDecomposition;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.linear.SingularValueDecomposition;
import org.apache.commons.math3.stat.StatUtils;
import org.apache.commons.math3.stat.descriptive.moment.SecondMoment;
import org.apache.commons.math3.stat.descriptive.moment.Variance;
import org.apache.commons.math3.stat.regression.OLSMultipleLinearRegression;

/**
 * <p>Implements weight least squares (WLS) to estimate the parameters of a
 * multiple linear regression model.</p> 
 * The WOLS is the same as the OLS but the observations are scaled by the weights.
 * </br>
 * </br>01/08/2012
 * </br>
 * @author Dr. Vlasios Voudouris, Daniil Kiose, Prof. Mikis Stasinopoulos and Dr Robert Rigby
 *
 */
 
 
public class WLSMultipleLinearRegression extends OLSMultipleLinearRegression 
{
	 /** original X sample data. */
	 public RealMatrix X;
	 /** the weighted X sample data. */
	 private RealMatrix xMatrix;
	 /** original Y sample data. */
	 private RealVector y;
	 /** the weighted Y sample data. */
	 private RealVector yVector;
	 /** the beta of the regression */
	 private RealVector beta;
	 /** Cached SVD decomposition of X matrix */
	 private SingularValueDecomposition svg;
	 /** Cached QR decomposition of X matrix */
	 private QRDecomposition qr = null;
	 /** whether to copy the original data */
	 private boolean copyOriginal=true;

	 
	/**
	 * <p>Implements weighted least squares (WLS) to estimate the parameters of a
	 * multiple linear regression model.</p> The WOLS is the same as the OLS but the observations are scaled by the weights.
	 * @author Vlasios Voudouris, Daniil Kiose and Mikis Stasinopoulos	
	 * ABM Analytics Ltd.  
	 */
	public WLSMultipleLinearRegression(boolean copyOriginal)
	{
		super();
		this.copyOriginal=copyOriginal;
	}
	
	/**
	 * <p>Implements weight ordinary least squares (WOLS) to estimate the parameters of a
	 * multiple linear regression model.</p> 
	 * The WOLS is the same as the OLS but the observations are scaled by the weights.
	 * @param y the [n,1] array representing the y sample
	 * @param x the [n,k] array representing design matrix
	 * @param w the [n,1] array representing the weights
	 */
	public WLSMultipleLinearRegression(ArrayRealVector y, BlockRealMatrix x, ArrayRealVector w, boolean noIntercept,boolean copyOriginal)
	{
		super();
		this.setNoIntercept(noIntercept);
		this.newSampleData(y,x,w);
		this.copyOriginal=copyOriginal;
	}
	
	
	/**
     * Loads model x and y sample data, overriding any previous sample. 
     * <b>Cloning is not performed to speed the operation</b>. 
     * Computes and caches QR decomposition of the X matrix.
	 * @param y the [n,1] array representing the y sample
	 * @param x the [n,k] array representing design matrix
	 * @param w the [n,1] array representing the weights
	 * @throws IllegalArgumentException if the x and y array data are not
     *             compatible for the regression
	 */
	 public void newSampleData(ArrayRealVector y, RealMatrix x, ArrayRealVector w) 
	 {
    	 	w.mapToSelf(new Sqrt());
		 	if(this.copyOriginal)
		 	{
		 		newSampleDataCopyOriginal(y,x,w);
		 	}
		 	else
		 	{
		 		newSampleDataNoCopy(y,x,w);
		 	}
	 }
	 
	 /**
	  * 
	  * @param y
	  * @param x
	  * @param w
	  */
	 private void newSampleDataCopyOriginal(ArrayRealVector y, RealMatrix x, ArrayRealVector w)
	 {
		 	this.y=y.copy(); //we need this for the fitted values and residuals.
		 	if (this.isNoIntercept())
	        {
		 		this.X=x.copy();

	        }
		 	else
		 	{
		 		setDesignMatrix(x);//add 1 for the Intercept;
		 	}		 			 	
		 	for(int row=0; row<x.getRowDimension();row++)
		 	{
		 		x.setRowVector(row,x.getRowVector(row).mapMultiplyToSelf(w.getEntry(row)));
		 	}
		 	//double[][] xw=x.getData();
		 	//double[] yw= y.ebeMultiply(w).getDataRef();
	        //validateSampleData(xw, yw); //we have already checked this in the gamlss algorithm. 
	        newYSampleData(y.ebeMultiply(w));
	        newXSampleData(x.getData(),w);
	 }
	 
	 /**
	  * 
	  * @param y
	  * @param x
	  * @param w
	  */
	 private void newSampleDataNoCopy(ArrayRealVector y, RealMatrix x, ArrayRealVector w)
	 {
		 	for(int row=0; row<x.getRowDimension();row++)
		 	{
		 		x.setRowVector(row,x.getRowVector(row).mapMultiplyToSelf(w.getEntry(row)));
		 	}
		 	//double[][] xw=x.getData();
		 	//double[] yw= y.ebeMultiply(w).getDataRef();
	        //validateSampleData(xw, yw); //we have already checked this in the gamlss algorithm. 
	        newYSampleData(y.ebeMultiply(w));
	        newXSampleData(x.getData(),w);
	 }
	 
	     /**
	     * Loads new y sample data, overriding any previous data.
	     *
	     * @param y the array representing the y sample
	     * @throws NullArgumentException if y is null
	     * @throws NoDataException if y is empty
	     */
	    protected void newYSampleData(ArrayRealVector y) {
	        if (y == null) {
	            throw new NullArgumentException();
	        }
	        if (y.getDimension() == 0) {
	            throw new NoDataException();
	        }
	        this.yVector = y;
	    }
	 
	     /**
	     * {@inheritDoc}
	     * <p>This implementation computes and caches the QR decomposition of the X matrix
	     * once it is successfully loaded.</p>
	     */
	    protected void newXSampleData(double[][] x, ArrayRealVector w) {
	    	 if (x == null) {
		            throw new NullArgumentException();
		        }
		        if (x.length == 0) {
		            throw new NoDataException();
		        }
		        if (this.isNoIntercept()) {
		            this.xMatrix = new Array2DRowRealMatrix(x, true);
		        } else { // Augment design matrix with initial unitary column
		            final int nVars = x[0].length;
		            final double[][] xAug = new double[x.length][nVars + 1];
		            for (int i = 0; i < x.length; i++) {
		                if (x[i].length != nVars) {
		                    throw new DimensionMismatchException(x[i].length, nVars);
		                }
		                xAug[i][0] = w.getEntry(i);
		                System.arraycopy(x[i], 0, xAug[i], 1, nVars);
		            }
		            this.xMatrix = new Array2DRowRealMatrix(xAug, false);
		        }   
	        this.qr = new QRDecomposition(getX());
	    }
	
	    
	 /**
	  * Add a column with 1s for the Intercept. 
	  * @param X the design matrix
	  */	    
	 private void setDesignMatrix(RealMatrix X)
	 {
		 double[][]x = X.getData();
		 final int nVars = x[0].length;
         final double[][] xAug = new double[x.length][nVars + 1];
         for (int i = 0; i < x.length; i++) {
             if (x[i].length != nVars) {
                 throw new DimensionMismatchException(x[i].length, nVars);
             }
             xAug[i][0] = 1.0d;
             System.arraycopy(x[i], 0, xAug[i], 1, nVars);
         }
         this.X = new Array2DRowRealMatrix(xAug, false);
	 }
	 
	    /**
	     * @return the weighted X sample data.
	     */
	     @Override
	    protected RealMatrix getX() {
	        return this.xMatrix;
	    }

	    /**
	     * @return the weighted Y sample data.
	     */
	    @Override
	    protected RealVector getY() {
	        return this.yVector;
	    }
	 
	 
	 /**
	  * This method assumes that the y, X and w has been already specified either
	  * by calling the <code>newSampleData<code> method or the relevant constructor  
	  * @return ArrayRealVector the fitted values or null if not applicable
	  */
	 public RealVector calculateFittedValues()
	 {
		 	if(this.copyOriginal)
		 	{
		 		return  this.X.operate(beta); // Xb
		 	}
		 	return null;
	 }      
	 
	 /**
	  * Calculates the fitted values without SAVING the 'beta' for later use
	  * @param isSVD whether to use SVD or QR for the design matrix X
	  * Note that SVD is more accurate but slower. 
	  * @return the fitted values or null if not applicable
	  */
	 public RealVector calculateFittedValues(boolean isSVD)
	 {
	    if(this.copyOriginal)
	    { 
		 if (!isSVD)
		 {
			return  this.X.operate(this.calculateBeta());
			
		 }
		 else
		 {
			 SingularValueDecomposition svg = new SingularValueDecomposition(getX());

			 return this.X.operate(svg.getSolver().solve(getY()));
		 }
		}
		return null;
	 }
	 
	 /**
	 * Calculates the regression coefficients using OLS.
	 *
	 * @return beta
	 */
	 @Override
	 public RealVector calculateBeta() {
	        beta= this.qr.getSolver().solve(getY());
	        return  beta;
	 }
	 
	 /**
	 * Calculates the regression coefficients using OLS using singular value decomposition.
	 * 
	 * @return beta
	 */
	 public RealVector calculateBetaSVD() 
	 {
	        svg = new SingularValueDecomposition(getX());
            beta= svg.getSolver().solve(getY());
	        return beta;
	  }
	 
	 /**
	 * {@inheritDoc}
	 * return null if not applicable
	 */
	 @Override
	 public double[] estimateResiduals() 
	 {
		 if(this.copyOriginal)
		 {
	        RealVector e = this.y.subtract(this.X.operate(beta));
	        return e.toArray();
		 }
		 return null;
	 }
	 
	 
	 /**
	 * {@inheritDoc}
	 * Double.NaN is not applicable
	 */
	 @Override
	 public double calculateTotalSumOfSquares() {
		if(this.copyOriginal)
		{ 
	        if (isNoIntercept()) {
	            return StatUtils.sumSq(this.y.toArray());
	        } else {
	            return new SecondMoment().evaluate(this.y.toArray());
	        }
		 }
		return Double.NaN;
	    }

	 /**
	 * {@inheritDoc}
	 * returns Double.NaN if no applicable 
	 */
	 @Override
	 public double calculateResidualSumOfSquares() {
		 if(this.copyOriginal)
		 {
	        final RealVector residuals = calculateResiduals();
	        return residuals.dotProduct(residuals);
		 }
		 return Double.NaN;
	    }
	 
	 
	 /**
	 * {@inheritDoc}
	 * returns null if no applicable
	 */
	 @Override
	 public RealVector calculateResiduals() {
		 if(this.copyOriginal)
		 {
	        return this.y.subtract(this.X.operate(beta));
		 }
		 return null;
	    }	
	 
	 /**
	 * {@inheritDoc}
	 * returns null if no applicable
	 */
	 @Override
	    public double[][] estimateRegressionParametersVariance() {
		 if(this.copyOriginal)
		 {
	        return calculateBetaVariance().getData();
		 }
		 return null;
	    }
	 
	 
	 /**
	     * <p>Calculates the variance-covariance matrix of the regression parameters.
	     * </p>
	     * <p>Var(b) = (X<sup>T</sup>X)<sup>-1</sup>
	     * </p>
	     * <p>Uses QR decomposition to reduce (X<sup>T</sup>X)<sup>-1</sup>
	     * to (R<sup>T</sup>R)<sup>-1</sup>, with only the top p rows of
	     * R included, where p = the length of the beta vector.</p>
	     *
	     * @return The beta variance-covariance matrix or null if not applicable
	     */
	    @Override
	    protected RealMatrix calculateBetaVariance() {
	       if(this.copyOriginal)
		   {
	        int p = getX().getColumnDimension();
	        RealMatrix Raug = qr.getR().getSubMatrix(0, p - 1 , 0, p - 1);
	        RealMatrix Rinv = new LUDecomposition(Raug).getSolver().getInverse();
	        return Rinv.multiply(Rinv.transpose());
		   }
	       return null;
	    }
	    
	    /**
	     * Calculates the variance of the y values.
	     *
	     * @return Y variance or Double.NaN if no applicable
	     */
	    @Override
	    protected double calculateYVariance() {
	    	if(copyOriginal)
		    {
	         return new Variance().evaluate(y.toArray());
		    }
	    	return Double.NaN;	
	    }
	 
	    /**
	     * <p>Calculates the variance of the error term.</p>
	     * Uses the formula <pre>
	     * var(u) = u &middot; u / (n - k)
	     * </pre>
	     * where n and k are the row and column dimensions of the design
	     * matrix X.
	     *
	     * @return error variance estimate or Double.NaN if no applicable
	     * @since 2.2
	     */
	    @Override
	    protected double calculateErrorVariance() {
	     if(copyOriginal)
	     {
	        RealVector residuals = calculateResiduals();
	        return residuals.dotProduct(residuals) /
	               (X.getRowDimension() - X.getColumnDimension());
	     }
	     return Double.NaN;	
	    } 
	    	    
}