/*
  Copyright 2012 by Dr. Vlasios Voudouris and ABM Analytics Ltd
  Licensed under the Academic Free License version 3.0
  See the file "LICENSE" for more information
  
  Dr. Vlasios Voudouris, Daniil Kiose, Prof. Mikis Stasinopoulos and Dr Robert Rigby
*/
package gamlss.distributions;

import gamlss.utilities.Controls;
import gamlss.utilities.MakeLinkFunction;
import gamlss.utilities.NormalDistr;

import java.util.Hashtable;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.util.FastMath;
import org.apache.commons.math3.distribution.NormalDistribution;
import org.apache.commons.math3.stat.descriptive.moment.*;

/**
 * 01/08/2012.
 * @author Dr. Vlasios Voudouris, Daniil Kiose, Prof.
 *  Mikis Stasinopoulos and Dr Robert Rigby.
 *
 */
public class NO extends NormalDistribution implements GAMLSSFamilyDistribution {
	/** Number of distribution parameters. */
	private final int numDistPar = 2;
	/** Hashtable to hold vectors of distribution parameters (mu, sigma). */
	private Hashtable<Integer, ArrayRealVector> distributionParameters
								   = new Hashtable<Integer, ArrayRealVector>();
	/** Hashtable to hold types of link functions for
	 *  the distribution parameters. */
	private Hashtable<Integer, Integer> distributionParameterLink 
										   = new Hashtable<Integer, Integer>();
	/** Object of the Normal distribution.  */
	private NormalDistr noDist;
	/** vector of values of mu distribution parameter. */
	private ArrayRealVector muV;
	/** vector of values of sigma distribution parameter. */
	private ArrayRealVector sigmaV;
	/** Temporary int for interim operations. */
	private int size;
	/** Temporary vector for interim operations. */
	private ArrayRealVector tempV;
	/** Temporary vector for interim operations. */
	private ArrayRealVector tempV2;
	
	/** This is the Gausian distribution with default 
	 * link (muLink="identity",sigmaLink="log"). */
	public NO() {
		
		this(DistributionSettings.IDENTITY, DistributionSettings.LOG);
	}
	
	/** 
	 * This is the Gausian distribution with supplied 
	 * link function for each of the distribution parameters.
	 * @param mulink - link function for mu
	 * @param sigmalink - link function for sigma
	 */
	public NO(final int mulink, final int sigmalink) {
		
		distributionParameterLink.put(DistributionSettings.MU, 
			   MakeLinkFunction.checkLink(DistributionSettings.NO, mulink));
		distributionParameterLink.put(DistributionSettings.SIGMA, 
			   MakeLinkFunction.checkLink(DistributionSettings.NO, sigmalink));
		noDist = new NormalDistr();
	}
	
	/** initializes the functions to calculate intial(starting).
	 *  values of distribution parameters 
	 *  @param y - response vector */
	public final void initialiseDistributionParameters(
													final ArrayRealVector y) {
		distributionParameters.put(DistributionSettings.MU, 
													setMuInitial(y));
		distributionParameters.put(DistributionSettings.SIGMA, 
												 	setSigmaInitial(y));
	}
	
	/** Calculates initial value of sigma.
	 * @param y - vector of values of response variable
	 * @return vector of initial values of sigma
	 */
	private ArrayRealVector setSigmaInitial(final ArrayRealVector y) {	
		tempV = new ArrayRealVector(y.getDimension());
		final double ySD  = new StandardDeviation().evaluate(y.getDataRef());	
		tempV.set(ySD);
		return tempV;

	}
	
	/**  Calculates initial value of mu, by assumption these values lie
 between observed data and the trend line.
	 * @param y - vector of values of response variable
	 * @return vector of initial values of mu
	 */
	private ArrayRealVector setMuInitial(final ArrayRealVector y) {
		size = y.getDimension();
		double[] out = new double[size];
		final double yMean = new Mean().evaluate(y.getDataRef());	
		for (int i = 0; i < size; i++) {
			out[i] = (y.getEntry(i) + yMean) / 2;
		}
		return new ArrayRealVector(out, false);
	}
	
	/** Calculates a first derivative of the likelihood function 
	 * in respect to supplied distribution parameter.
	 * @param whichDistParameter - distribution parameter
	 * @param y - vector of values of likelihood function
	 * @return vector of first derivative of the likelihood 
	 */
	public final ArrayRealVector firstDerivative(final int whichDistParameter,
										   		 final ArrayRealVector y) {	
			setInterimArrays(y);
			tempV = null;		
			switch (whichDistParameter) {
	        case DistributionSettings.MU:
	        	tempV = dldm(y);
	           break;
	        case DistributionSettings.SIGMA:
	        	tempV = dlds(y);
	           break;
	        default: 
				System.err.println("Requested first order "
						+ "derivative does not exist");
			   break;
			}
			return tempV;
		}
	
	/**
	 * Set tempV2 = sigma*sigma.
	 * @param y - response variable
	 */
	private void setInterimArrays(final ArrayRealVector y) {
		
		muV     = distributionParameters.get(DistributionSettings.MU);
		sigmaV  = distributionParameters.get(DistributionSettings.SIGMA);
		size = y.getDimension();
		tempV2 = sigmaV.ebeMultiply(sigmaV);
	}
	
	/** Calculates a second derivative of the likelihood function in
 	 * respect to supplied distribution parameter.
	 * @param whichDistParameter - distribution parameter
	 * @param y - vector of values of likelihood function 
	 * @return vector of second derivative of the likelihood 
	 */
	public final ArrayRealVector secondDerivative(final int whichDistParameter,
										    	  final ArrayRealVector y) {
		
				tempV = null;		
				switch (whichDistParameter) {
			       case DistributionSettings.MU:
			    	 tempV = d2ldm2();
			          break;
			     case DistributionSettings.SIGMA:
			    	 tempV = d2lds2();
			     break;
			        default: 
						System.err.println("Requested second order "
								+ "derivative does not exist");
					   break;
			    }
				return tempV;
	}	
	
	/** Calculates a second cross derivative of the likelihood function in
	 * respect to supplied distribution parameter.
	 * @param whichDistParameter1 - first distribution parameter
	 * @param whichDistParameter2 - second distribution parameter
	 * @param y - vector of values of likelihood function 
	 * @return vector of cross derivatives of the likelihood 
	 */
	public final ArrayRealVector secondCrossDerivative(
			 									 final int whichDistParameter1,
												 final int whichDistParameter2, 
												 final ArrayRealVector y) {		
				tempV = null;		
				if (whichDistParameter1 == DistributionSettings.MU) {
					switch (whichDistParameter2) {
					case DistributionSettings.SIGMA:
						tempV = d2ldmdd(y);                      
						break;
			          default: 
				  			System.err.println("Second derivative"
				  					+ " does not exist");
				  		break;
					}
				}
				return tempV;
		}
	
	/** Second cross derivative of likelihood function in
	 *  respect to mu and sigma (d2ldmdd = d2l/dmu*dsigma).
	 * @param y - vector of values of response variable
	 * @return  a vector of Second cross derivative
	 */
	private ArrayRealVector d2ldmdd(final ArrayRealVector y) {
		  //d2ldmdd = function(y) rep(0,length(y) 
		  return new ArrayRealVector(y.getDimension());
		  // all elemnts are 0's);
	}

		/**  First derivative dldm = dl/dmu, 
		 * where l - log-likelihood function.
		 * @param y - vector of values of response variable
		 * @return  a vector of first derivative dldm = dl/dmu
		 */
		private ArrayRealVector dldm(final ArrayRealVector y) {
			  double[] out = new double[size];
			  for (int i = 0; i < size; i++) {
				  //(1/sigma^2)*(y-mu)  
				  out[i] = (1 / tempV2.getEntry(i)) 
						     * (y.getEntry(i) - muV.getEntry(i));
			  }
			  return  new ArrayRealVector(out, false);
		}
		
	/** Second derivative d2ldm2= (d^2l)/(dmu^2), 
	 * where l - log-likelihood function.
	 * @return  a vector of Second derivative d2ldm2= (d^2l)/(dmu^2)
	 */
	private ArrayRealVector d2ldm2() {
		double[] out = new double[size];
		for (int i = 0; i < size; i++) {
			//-(1/sigma^2)
			out[i] = -(1 / tempV2.getEntry(i));
		}
		return  new ArrayRealVector(out, false);
	}
	
	/** First derivative dlds = dl/dsigma, 
	 * where l - log-likelihood function.
	 * @param y - vector of values of response variable
	 * @return  a vector of First derivative dlds = dl/dsigma
	 */
	private ArrayRealVector dlds(final ArrayRealVector y) {	
		double[] out = new double[size];		
		for (int i = 0; i < size; i++) {
			//((y-mu)^2-sigma^2)/(sigma^3)
			out[i] = (FastMath.pow(y.getEntry(i) - muV.getEntry(i), 2)
				   - tempV2.getEntry(i)) / FastMath.pow(sigmaV.getEntry(i), 3);
		}	
		return  new ArrayRealVector(out, false);
	}
	
	/** Second derivative d2lds2= (d^2l)/(dsigma^2).
	 * @return  a vector of second derivative d2lds2= (d^2l)/(dsigma^2)
	 */
	private ArrayRealVector d2lds2() {
		//d2ldd2 = function(sigma) -(2/(sigma^2)),
		double[] out = new double[size];		
		for (int i = 0; i < size; i++) {
			//out[i] = -(2/FastMath.pow(sigma.getEntry(i), 2));
			out[i] = -(2 / tempV2.getEntry(i));
		}	
		return  new ArrayRealVector(out, false);
	}
	
	/** Computes the global Deviance Increament.
	 * @param y - vector of response variable values
	 * @return vector of global Deviance Increament values 
	 */
	public final ArrayRealVector globalDevianceIncreament(
													final ArrayRealVector y) {

		size = y.getDimension();
		double[] out = new double[size];

		double[] muArr = distributionParameters.get(
									DistributionSettings.MU).getDataRef();
		double[] sigmaArr = distributionParameters.get(
									DistributionSettings.SIGMA).getDataRef();
		for (int i = 0; i < size; i++) {
			
			out[i] = (-2) * dNO(y.getEntry(i), muArr[i], sigmaArr[i], 
													Controls.LOG_LIKELIHOOD);
		}
		return  new ArrayRealVector(out, false);
	}
	
	/**
	 * Computes the probability density function (PDF) 
	 * of this distribution evaluated at the specified point x.
	 * @param x - value of response variable
	 * @param mu - value of mu distribution parameter
	 * @param sigma - value of sigma distribution parameter
	 * @param isLog  - logical, whether to take log of the function or not
	 * @return value of probability density function
	 */
	public final double dNO(final double x, 
					                 final double mu, 
					                 final double sigma, 
					                 final boolean isLog) {
		
		//fy <- dnorm(x, mean=mu, sd=sigma, log=log)
		noDist.setDistrParameters(mu, sigma);
		if (isLog) {

			return FastMath.log(noDist.density(x));
		} else {
			
			return noDist.density(x);
		}
	}
	
	/**
	 * Computes the probability density function (PDF) 
	 * of this distribution evaluated at the specified point x.
	 * default (mu = 0, sigma = 1)
	 * @param x - value of response variable 
	 * @return value of probability density function
	 */
	public final double dNO(final double x) {
		return dNO(x, 0.0, 1.0, false);
	}

	/** Computes the cumulative distribution function
	 *  P(X <= q) for a random variable X.
	 * whose values are distributed according to this distribution
	 * @param q - vector of quantiles
	 * @param mu - vector of mu distribution parameter values
	 * @param sigma - vector of sigma distribution parameter values
	 * @param lowerTail - logical, if TRUE (default), probabilities
	 *  are P[X <= x] otherwise, P[X > x].
	 * @param islog - logical, if TRUE, probabilities p are given as log(p).
	 * @return vector of cumulative probability function values P(X <= q)
	 */
	public final double pNO(final double q, 
			                final double mu, 
			                final double sigma, 
			                final boolean lowerTail, 
			                final boolean islog) {
		
		// {  if (any(sigma <= 0))stop(paste("sigma must be positive"))
		if (sigma <= 0) {
			System.err.println("sigma must be positive");
			return -1.0;
		}
		
		noDist.setDistrParameters(mu, sigma);
		noDist.cumulativeProbability(q);
		double out = noDist.cumulativeProbability(q);		    
	
	    if (!lowerTail) {
	    	
	    	if (islog) {
	    		
	    		out = FastMath.log(1 - out);
	    	} else {
	    		
	    		out = 1 - out;
	    	}
	    } else if (islog) {
	    	
	    	out = FastMath.log(out);
	    }
	    return out;
	  }

	/** Computes the cumulative distribution function
	 *  P(X <= q) for a random variable X.
	 * whose values are distributed according to this distribution
	 * default (mu = 0, sigma = 1)
	 * @param q - quantile
	 * @return value of cumulative probability function P(X <= q)
	 */
	public double pNO(final double q) {
		return pNO(q, 0.0, 1.0, true, false);
	}

	/** Computes the quantile (inverse cumulative probability)
	 *  function  of this distribution.
	* @param p - value of cumulative probability
	* @param mu -  value of mu distribution parameters
	* @param sigma -  value of sigma distribution parameters
	* @param lowerTail - logical; if TRUE (default), 
	* probabilities are P[X <= x] otherwise, P[X > x].
	* @param logp - logical; if TRUE, probabilities p are given as log(p).
	* @return value of quantile function
	*/
	public final double qNO(final double p, 
							final double mu, 
							final double sigma, 
							final boolean lowerTail, 
							final boolean logp) {
		
		// {  if (any(sigma <= 0))stop(paste("sigma must be positive"))
		if (sigma <= 0) {
			System.err.println("sigma must be positive");
			return -1.0;
		}
		
		double out = p;
		if (logp) {
			
			out = FastMath.exp(out);
		}
		
		if (out < 0 || out > 1) {
			
			System.err.println("Error: p must be between 0 and 1");
		}	
	    if (!lowerTail) {
	    	 out = 1 - out;
	    }
	    noDist.setDistrParameters(mu, sigma);
		return noDist.inverseCumulativeProbability(out);
	}
	
	/** Computes the quantile (inverse cumulative probability)
	 *  function  of this distribution.
	* default (mu = 0, sigma = 1)
	* @param p - value of cumulative probability 
	* @return value of quantile function
	*/
	public double qNO(final double p) {
		return qNO(p, 0.0, 1.0, true, false);
	}
	

	/** Generates a random value from this distribution.
	 * @param mu -  value of mu distribution parameters
	 * @param sigma -  value of sigma distribution parameters
	 * @return random value
	 */	
	private double rNO(final double mu, 
			           final double sigma) {
		
		noDist.setDistrParameters(mu, sigma);
		return  noDist.sample();
	}	
	
	
	/** Generates a random value from this distribution.
	 * default (mu = 0, sigma = 1)
	 * @return random value 
	 */	
	private double rNO() {
		
		return rNO(0.0, 1.0);
	}

	
	 /**
	  * Extends vector x by copying its values
	  *  consequently until it reaches dimension maxDim.
	  * @param x - vector of quantiles
	  * @param xDim - dimension of vector x
	  * @param maxDim - dimension of the larget vector of x, mu or sigma
	  * @return extended vector x
	  */
/*	  public ArrayRealVector extendX(ArrayRealVector x, int xDim, double maxDim)
	  {
		int n=0;			
		double [] xArr = x.getDataRef();
		double [] out = new double[(int)maxDim];		
		for (int i=0; i<(int)maxDim; i++){				
			out[i] = xArr[n];
			n++;
			if (n==xDim)
			{
				n=n-xDim;
			}				
		}			
		return new ArrayRealVector(out,false);
	  }	
	
	  /**
	   *  Extends vector mu by copying its values consequently 
	   *  until it reaches dimension maxDim 
	   * @param mu - vector of mu values
	   * @param muDim - dimension of vector mu
	   * @param maxDim - dimension of the larget vector of x, mu or sigma
	   * @return extended vector mu
	   */
/*	  public ArrayRealVector extendMu(ArrayRealVector mu, int muDim, double maxDim)
	  {
		int n=0;			
		double [] muArr = mu.getDataRef();
		double [] out = new double[(int)maxDim];		
		for (int i=0; i<(int)maxDim; i++){				
			out[i] = muArr[n];
			n++;
			if (n==muDim)
			{
				n=n-muDim;
			}				
		}			
		return new ArrayRealVector(out,false);
	  }
	

	/**
	 * Extends vector sigma by copying its values 
	 * consequently until it reaches dimension maxDim 
	 * @param sigma - vector of sigma values
	 * @param sigmaDim -  dimension of vector sigma
	 * @param maxDim - dimension of the larget vector of x, mu or sigma
	 * @return extended vector sigma
	 */
/*	  private ArrayRealVector extendSigma(ArrayRealVector sigma, 
 * int sigmaDim, double maxDim)
	  {
		int n=0;			
		double [] sigmaArr = sigma.getDataRef();
		double [] out = new double[(int)maxDim];		
		for (int i=0; i<(int)maxDim; i++){				
			out[i] = sigmaArr[n];
			n++;
			if (n==sigmaDim)
			{
				n=n-sigmaDim;
			}				
		}			
		return new ArrayRealVector(out,false);
	  }
*/
	
	/** Checks whether entries of ArrayRealVector y are valid.
	 * @param y - vector of values of response variable
	 * @return boolean
	 */
	public final boolean isYvalid(final ArrayRealVector y) {
		return true;
	}
	
	/** Checks whether entries of ArrayRealVectors 
	 * of distribution parameters are valid.
	* @param whichDistParameter - distribution parameter
	  @return Hashtable of booleans
	 */
	public final boolean areDistributionParametersValid(
												final int whichDistParameter) {
		boolean  out =  false;
		switch (whichDistParameter) {
      case DistributionSettings.MU:
      	 out = isMuValid(distributionParameters.get(
      			 								DistributionSettings.MU));
         break;
      case DistributionSettings.SIGMA:
     	     out = isSigmaValid(distributionParameters.get(
     	    		 							DistributionSettings.SIGMA));
        break;
		default: System.out.println("The specific distribution parameter"
				+ " does not exist for this distribution");
			break;
		}
		return out;
	}	

	/**
	 * Checks whether the mu distribution parameter is valid.
	 * @param mu - vector of mu (mean) values
	 * @return - boolean
	 */
	private boolean isMuValid(final ArrayRealVector mu) {
		return true;
	}

	/**
	 * Checks whether the sigma distribution parameter is valid.
	 * @param sigma - vector of sigma (standard deviation) values
	 * @return - - boolean
	 */
	private boolean isSigmaValid(final ArrayRealVector sigma) {
		return sigma.getMinValue() > 0;	
	}			
	

	/** get the number of distribution parameters
	 *  (currently up-to 4 distribution parameters). 
	 *  @return number of distribution parameters*/
	public final int getNumberOfDistribtionParameters() {
		return numDistPar;
	}
	
	/** get the type (e.g., continuous) of distribution.
	 * @return  type of distributuion  (Continuous, Discrete or Mixed)*/
	public final int getTypeOfDistribution() {
		return DistributionSettings.CONTINUOUS;
	}

	/** get family (e.g., Normal) of distribution. 
	 * @return distribution name*/
	public final int getFamilyOfDistribution() {
		return DistributionSettings.NO;
	}
	
	/**
	 * Set distribution parsameters.
	 * @param whichDistParameter - the fitting distribution parameter
	 * @param fvDistributionParameter - vector of values of 
	 * fitting distribution parameter
	 */
	public final void setDistributionParameter(final int whichDistParameter, 
			   				final ArrayRealVector fvDistributionParameter) {
		
		this.distributionParameters.put(whichDistParameter,
												fvDistributionParameter);
	}
	
	/**
	 * Get distribution parsameters.
	 * @param whichDistParameter - distribution parameter
	 * @return - vector of distribution parameter values
	 */
	public final ArrayRealVector getDistributionParameter(
											final int whichDistParameter) {
		return this.distributionParameters.get(whichDistParameter);
	}
	
	/** Get the link function type of the current distribution parameter.
	 * @param whichDistParameter - distribution parameter
	 * @return link function type
	 */
	public final int getDistributionParameterLink(
												final int whichDistParameter) {
		return distributionParameterLink.get(whichDistParameter);
	}
}
