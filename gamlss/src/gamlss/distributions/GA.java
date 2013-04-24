/*
  Copyright 2012 by Dr. Vlasios Voudouris and ABM Analytics Ltd
  Licensed under the Academic Free License version 3.0
  See the file "LICENSE" for more information
*/
package gamlss.distributions;

import gamlss.utilities.Controls;
import gamlss.utilities.GammaDistr;
import gamlss.utilities.MakeLinkFunction;
import java.util.Hashtable;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.special.Gamma;
import org.apache.commons.math3.stat.descriptive.moment.Mean;
import org.apache.commons.math3.util.FastMath;
import org.apache.commons.math3.distribution.UniformRealDistribution;

	/**
	 * @author Dr. Vlasios Voudouris, Daniil Kiose, 
	 * Prof. Mikis Stasinopoulos and Dr Robert Rigby.
	 */
	public class GA implements GAMLSSFamilyDistribution {
	
	/** Number of distribution parameters. */
	private final int numDistPar = 2;
	/** Hashtable to hold vectors of distribution
	 *  parameters (mu, sigma, ...). */
	private Hashtable<Integer, ArrayRealVector> distributionParameters 
	= new Hashtable<Integer, ArrayRealVector>();
	/** Hashtable to hold types of link functions for
	 *  the distribution parameters.*/
	private Hashtable<Integer, Integer> distributionParameterLink 
	= new Hashtable<Integer, Integer>();
	/** Object of the Gamma distribution .*/
	private GammaDistr gammaDistr;
	/** Temporary vector for interim operations. */
	private ArrayRealVector tempV;
	/** Temporary int for interim operations. */
	private int size;
	/** vector of values of mu distribution parameter. */
	private ArrayRealVector muV;
	/** vector of values of sigma distribution parameter. */
	private ArrayRealVector sigmaV;
	/** Temporary vector for interim operations. */
	private ArrayRealVector tempV2;

	/** This is the Gamma distribution with default
	 *  link (muLink="log",sigmaLink="log") .*/
	public GA() {
		
		this(DistributionSettings.LOG, DistributionSettings.LOG);
	}
	
	/** This is the Gamma distribution with supplied 
	 * link function for each of the distribution parameters.
	 * @param muLink - link function for mu distribution parameter
	 * @param sigmaLink - link function for sigma distribution parameter */
	public GA(final int muLink, final int sigmaLink) {
		
		distributionParameterLink.put(DistributionSettings.MU, 
				MakeLinkFunction.checkLink(DistributionSettings.GA, muLink));
		distributionParameterLink.put(DistributionSettings.SIGMA,  
				MakeLinkFunction.checkLink(DistributionSettings.GA, sigmaLink));
		
		gammaDistr = new GammaDistr();
	}


	/** initializes the functions to calculate intial(starting) 
	 * values of distribution parameterss. */
	public final void initialiseDistributionParameters(
												final ArrayRealVector y) {
	    
		distributionParameters.put(DistributionSettings.MU, 
				setMuInitial(y));
		distributionParameters.put(DistributionSettings.SIGMA, 
				setSigmaInitial(y));
	}
	
	/** Calculates initial value of sigma.
	 * @param y - vector of values of response variable
	 * @return - a vector of  initial values of sigma
	 */
	private ArrayRealVector setSigmaInitial(final ArrayRealVector y) {	
		//sigma.initial = expression({sigma <- rep(1,length(y))}) ,
		tempV = new ArrayRealVector(y.getDimension());
		tempV.set(1);
		return tempV;
	}
	
	/**  Calculates initial value of mu, by assumption these
 	 * values lie between observed data and the trend line.
	 * @param y - vector of values of response variable
	 * @return  a vector of initial values of mu
	 */
	private ArrayRealVector setMuInitial(final ArrayRealVector y) {
		//mu.initial = expression({mu <- (y+mean(y))/2})
		size = y.getDimension();
		double[] out = new double[size];
		Mean mean = new Mean();	
		double yMean = mean.evaluate(y.getDataRef());	
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

	/**  First derivative dldm = dl/dmu,
	* where l - log-likelihood function.
	* @param y - vector of values of response variable
	* @return  a vector of first derivative dldm = dl/dmu
	*/
	private ArrayRealVector dldm(final ArrayRealVector y) {
		  //dldm = function(y,mu,sigma) (y-mu)/((sigma^2)*(mu^2)),
		  double[] out = new double[size];
		  for (int i = 0; i < size; i++) {
			  out[i] = (y.getEntry(i) - muV.getEntry(i)) 
			    		/ (tempV2.getEntry(i) * muV.getEntry(i) 
			    									* muV.getEntry(i));
		  }
		   return  new ArrayRealVector(out, false);
		}

	/** First derivative dlds = dl/dsigma,
	 * where l - log-likelihood function.
	 * @param y - vector of values of response variable
	 * @return  a vector of First derivative dlds = dl/dsigma
	 */
	private ArrayRealVector dlds(final ArrayRealVector y) {	
		//function(y,mu,sigma)  (2/sigma^3)*((y/mu)-log(y)
		//+log(mu)+log(sigma^2)-1+digamma(1/(sigma^2))),
		double[] out = new double[size];		
		for (int i = 0; i < size; i++) {
			out[i] = (2 / (tempV2.getEntry(i) * sigmaV.getEntry(i))) 
							* ((y.getEntry(i) / muV.getEntry(i)) 
									- FastMath.log(y.getEntry(i)) 
										+ FastMath.log(muV.getEntry(i)) 
										+ FastMath.log(tempV2.getEntry(i)) 
								- 1 + Gamma.digamma(1 / (tempV2.getEntry(i))));
		}	
		return  new ArrayRealVector(out, false);
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

	/** Second derivative d2ldm2= (d^2l)/(dmu^2), 
	 * where l - log-likelihood function.
	 * @return  a vector of Second derivative d2ldm2= (d^2l)/(dmu^2)
	 */
	private ArrayRealVector d2ldm2() {
		//d2ldm2 = function(mu,sigma) -1/((sigma^2)*(mu^2))
		double[] out = new double[size];
		for (int i = 0; i < size; i++) {
			out[i] = -1 / (tempV2.getEntry(i) 
					* (muV.getEntry(i) * muV.getEntry(i)));
		}
		return  new ArrayRealVector(out, false);
	}

	/** Second derivative d2lds2= (d^2l)/(dsigma^2),
	 * where l - log-likelihood function.
	 * @return  a vector of First derivative d2lds2= (d^2l)/(dsigma^2)
	 */
	private ArrayRealVector d2lds2() {
		//d2ldd2 = function(sigma) (4/sigma^4)
		//-(4/sigma^6)*trigamma((1/sigma^2)),
		double[] out = new double[size];		
		for (int i = 0; i < size; i++) {
			out[i] = (4 / (tempV2.getEntry(i)  * tempV2.getEntry(i))) 
						- (4 / (tempV2.getEntry(i) * tempV2.getEntry(i) 
								* tempV2.getEntry(i))) * Gamma.trigamma(1 
													   / (tempV2.getEntry(i)));
		}	
		return  new ArrayRealVector(out, false);
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

	/** Second cross derivative.
	 * @param y - vector of values of response variable
	 * @return  a vector of Second cross derivative
	 */
	private ArrayRealVector d2ldmdd(final ArrayRealVector y) {	  
		  //d2ldmdd = function(y)  rep(0,length(y)),
		  // all elemnts are 0's
		  return new ArrayRealVector(y.getDimension());
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
			
			out[i] = (-2) * dGA(y.getEntry(i), muArr[i], sigmaArr[i], 
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
	public double dGA(final double x, 
            					final double mu, 
            					final double sigma, 
            					final boolean isLog) {
		//if (any(mu <= 0))  stop(paste("mu must be positive", "\n", ""))
		if (mu <= 0) {
			System.err.println("mu must be positive");
			return -1.0;
		}
		//if (any(sigma <= 0))  stop(paste("sigma must be positive", "\n", ""))
		if (sigma <= 0) {
			System.err.println("sigma must be positive");
			return -1.0;
		}
		//if (any(x < 0))  stop(paste("x must be positive", "\n", ""))
		if (x < 0) {
			System.err.println("y must be positive");
			return -1.0;
		}
		
		//log.lik <- (1/sigma^2)*log(x/(mu*sigma^2))
		//-x/(mu*sigma^2)-log(x)-lgamma(1/sigma^2
		double out = (1 / (sigma * sigma)) * FastMath.log(x 
						/ (mu * sigma * sigma)) - x / (mu * sigma * sigma)
					  - FastMath.log(x) - Gamma.logGamma(1 / (sigma * sigma));
		if (!isLog) {
			//if(log==FALSE) fy  <- exp(log.lik) else fy <- log.lik
			out = FastMath.exp(out);
		}
		return  out;
	}
	

	/**
	 * dGA(x) launches dGA(x, mu, sigma, nu, isLog) 
	 * with deafult mu=1, sigma=1, isLog=false.
	 * @param x - vlaue of response variable
	 * @return value of probability density function
	 */
	public double  dGA(final double x) {
		//dGA<-function(x, mu=1, sigma=1, log=FALSE)
		return dGA(x, 1.0, 1.0, false);
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
	public double pGA(final double q, 
                       final double mu, 
                       final double sigma, 
                       final boolean lowerTail, 
                       final boolean islog) {

		//if (any(mu <= 0))  stop(paste("mu must be positive", "\n", ""))
		if (mu <= 0) {
			System.err.println("mu must be positive");
			return -1.0;
		}
		//if (any(sigma <= 0))  stop(paste("sigma must be positive", "\n", ""))
		if (sigma <= 0) {
			System.err.println("sigma must be positive");
			return -1.0;
		}
		//if (any(q < 0))  stop(paste("y must be positive", "\n", "")) 
		if (q < 0) {
			System.err.println("y must be positive");
			return -1.0;
		}

		double out = 0;		
		//cdf <- pgamma(q,shape=1/sigma^2,scale=mu*sigma^2, 
		//lower.tail = lower.tail, log.p = log.p)
		final double temp = sigma * sigma;
		gammaDistr.setDistrParameters(1 / temp, mu * temp);
		out = gammaDistr.cumulativeProbability(q);
		if (!lowerTail) {
		    if (islog) {
		    	out = FastMath.log(1 - out);
		    } else {
		    	out = 1 - out;
		    	}
		    } else if (islog) {
		    	out = FastMath.log(out);
		    }
		return  out;
	}

	/**
	 * pGA(q) launches pGA(q, mu, sigma, isLog) with deafult mu=1, sigma=1, 
	 * loweTail = true, isLog = false.
	 * @param q - quantile
	 * @return value of cumulative probability function P(X <= q)
	 */
	public double pGA(final double q) {
		//pGA <- function(q, mu=1, sigma=1, lower.tail = TRUE, log.p = FALSE)
		return pGA(q, 1.0, 1.0, true, false);
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
	public double qGA(final double p, 
					   final double mu, 
					   final double sigma, 
					   final boolean lowerTail, 
					   final boolean logp) {
		//if (any(mu <= 0))  stop(paste("mu must be positive", "\n", ""))
		if (mu <= 0) {
			System.err.println("mu must be positive");
			return -1.0;
		}
		//if (any(sigma <= 0))  stop(paste("sigma must be positive", "\n", ""))
		if (sigma <= 0) {
			System.err.println("sigma must be positive");
			return -1.0;
		}	

		//if (any(p < 0)|any(p > 1))  
		//stop(paste("p must be between 0 and 1", "\n", ""))
		if (p < 0 || p > 1) {
			System.err.println("p must be between 0 and 1");
			return -1;
		}
	
		//q <- qgamma(p,shape=1/sigma^2,scale=mu*sigma^2, 
		//lower.tail = lower.tail, log.p = log.p)
		final double temp = sigma * sigma;
		gammaDistr.setDistrParameters(1 / temp, mu * temp);
		
		double out = 0;	
		if (!lowerTail) {
			
		    if (logp) {
		    	
		    	out = gammaDistr.inverseCumulativeProbability(1
		    										- FastMath.log(p));
		    } else {
		    	
		    	out = gammaDistr.inverseCumulativeProbability(1 - p);
		    	}
		} else if (logp) {
			
		    out = gammaDistr.inverseCumulativeProbability(FastMath.log(p));
		} else {
			
		    out = gammaDistr.inverseCumulativeProbability(p);
		}
		return  out;
	}

	/**
	 * qGA(p) launches qGA(p, mu, sigma, lowerTail, isLog)
	 *  with deafult mu=1, sigma=1.
	 * lowerTail = true, isLog = false.
	* @param p - value of cumulative probability
	 * @return quantile
	 */
	//qGA <- function(p, mu=1, sigma=1,  lower.tail = TRUE, log.p = FALSE
	public final double qGA(final double p) {
		return qGA(p, 1.0, 1.0, true, false);
	}
	

	/** Generates a random value from this distribution.
	 * @param mu -  value of mu distribution parameters
	 * @param sigma -  value of sigma distribution parameters
	 * @param uDist -  object of UniformRealDistribution class;
	 * @return random value
	 */	
	private double rGA(final double mu, 
	           		   final double sigma,
	           		   final UniformRealDistribution uDist) {
		//r <- qST3(p,mu=mu,sigma=sigma,nu=nu,tau=tau)
		//p <- runif(n)
		return qGA(uDist.sample(), mu, sigma, true, false);
	}

	/**
	* rGA(n) launches rGA(n, mu, sigma, nu,  tau) 
	* with deafult mu=1, sigma=1.
	* @param uDist -  object of UniformRealDistribution class;
	* @return random value 
	*/
	//rGA <- function(n, mu=1, sigma=1)
	public final double rGA(final UniformRealDistribution uDist) {
		return rGA(1.0, 1.0, uDist);
	}

	/** Checks whether entries of ArrayRealVector y are valid.
	 * @param y - vector of values of response variable
	 * @return boolean
	 */
	public final boolean isYvalid(final ArrayRealVector y) {
		return y.getMinValue() > 0;
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
	 * Checks whether the mu distribution parameter is valid
	 * @param mu - vector of mu (mean) values
	 * @return - boolean
	 */
	private boolean isMuValid(final ArrayRealVector mu) {
		return mu.getMinValue() > 0;
	}


	/**
	 * Checks whether the sigma distribution parameter is valid
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
		return DistributionSettings.GA;
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
