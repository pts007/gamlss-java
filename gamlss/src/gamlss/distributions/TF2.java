/*
  Copyright 2012 by Dr. Vlasios Voudouris and ABM Analytics Ltd
  Licensed under the Academic Free License version 3.0
  See the file "LICENSE" for more information
*/
package gamlss.distributions;

import gamlss.utilities.Controls;
import gamlss.utilities.MakeLinkFunction;
import gamlss.utilities.NormalDistr;
import gamlss.utilities.TDistr;

import java.util.Hashtable;

import org.apache.commons.math3.distribution.TDistribution;
import org.apache.commons.math3.distribution.UniformRealDistribution;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.special.Gamma;
import org.apache.commons.math3.stat.descriptive.moment.Mean;
import org.apache.commons.math3.stat.descriptive.moment.StandardDeviation;
import org.apache.commons.math3.util.FastMath;

/**
 * @author Dr. Vlasios Voudouris, Daniil Kiose, 
 * Prof. Mikis Stasinopoulos and Dr Robert Rigby.
 */
public class TF2 implements GAMLSSFamilyDistribution {
																										/** Number of distribution parameters */
	/** Number of distribution parameters. */
	private final int numDistPar = 3;
	/** Hashtable to hold vectors of distribution 
	 * parameters values (mu, sigma, ...). */
	private Hashtable<Integer, ArrayRealVector> distributionParameters 
							= new Hashtable<Integer, ArrayRealVector>();
	/** Hashtable to hold types of link functions 
	 * for the distribution parameters. */
	private Hashtable<Integer, Integer> distributionParameterLink 
							= new Hashtable<Integer, Integer>();
	/** vector of values of mu distribution parameter. */
	private ArrayRealVector muV;
	/** vector of values of sigma distribution parameter. */
	private ArrayRealVector sigmaV;
	/** vector of values of nu distribution parameter. */
	private ArrayRealVector nuV;
	/** Temporary vector for interim operations. */
	private ArrayRealVector tempV;	
	/** Temporary vector for interim operations. */
	private ArrayRealVector tempV2;	
	/** Temporary vector for interim operations. */
	private ArrayRealVector sigma1;
	/** Temporary int for interim operations. */
	private int size;
	/** Object of TF Gamlss distribution . */
	private TF tf;
	/** Object of Normal distribution . */
	private NormalDistr noDist;
	/** Object of t distribution class. */
	private TDistr tdDist;
	/** Temporary array for interim operations. */
	private double[] ds1dv;
	/** Temporary array for interim operations. */
	private double[] ds1dd;
	


	/** This is the Student's t-distribution2with default 
	 * link (muLink="identity",sigmaLink="log", nuLink="log"). */
	public TF2() {
	
	this(DistributionSettings.IDENTITY, 
		 DistributionSettings.LOG, 
		 DistributionSettings.LOGSHIFTTO2);
	}
	
	/** This is the Student's t-distribution with supplied link 
	 * function for each of the distribution parameters.
	 * @param muLink - link function for mu distribution parameter
	 * @param sigmaLink - link function for sigma distribution parameter
	 * @param nuLink - link function for nu distribution parameter*/
	public TF2(final int muLink, 
			  final int sigmaLink, 
			  final int nuLink) {
	
		distributionParameterLink.put(DistributionSettings.MU, 
			MakeLinkFunction.checkLink(DistributionSettings.TF2, muLink));
		distributionParameterLink.put(DistributionSettings.SIGMA,  
			MakeLinkFunction.checkLink(DistributionSettings.TF2, sigmaLink));
		distributionParameterLink.put(DistributionSettings.NU,  
			MakeLinkFunction.checkLink(DistributionSettings.TF2, nuLink));
		
		tf = new TF();
		noDist = new NormalDistr();
		tdDist = new TDistr();
	}

	/** Initialises the distribution parameters.
	 * @param y - response variable */
	public final void initialiseDistributionParameters(
												final ArrayRealVector y) {
		
		distributionParameters.put(DistributionSettings.MU, 
													setMuInitial(y));
		distributionParameters.put(DistributionSettings.SIGMA, 
													setSigmaInitial(y));
		distributionParameters.put(DistributionSettings.NU, 
													setNuInitial(y));
	}
		
	/**  Calculate and set initial value of mu, by assumption 
	 * these values lie between observed data and the trend line.
	 * @param y - vector of values of response variable
	 * @return vector of initial values of mu
	 */
	private ArrayRealVector setMuInitial(final ArrayRealVector y) {
		//mu.initial =  expression(mu <- (y+mean(y))/2)
		size = y.getDimension();
		double[] out = new double[size];
		Mean mean = new Mean();	
		double yMean = mean.evaluate(y.getDataRef());	
		for (int i = 0; i < size; i++) {
			out[i] = (y.getEntry(i) + yMean) / 2;
		}
		return new ArrayRealVector(out, false);
	}
	
	/** Calculate and set initial value of sigma.
	 * @param y - vector of values of response variable
	 * @return vector of initial values of sigma
	 */
	private ArrayRealVector setSigmaInitial(final ArrayRealVector y) {
		//sigma.initial = expression(sigma<- rep(sd(y), length(y))),
		tempV = new ArrayRealVector(y.getDimension());
		final double out = new StandardDeviation().evaluate(y.getDataRef());
		tempV.set(out);
		return tempV;
	}
	
	/** Calculate and set initial value of nu.
	 * @param y - vector of values of response variable
	 * @return vector of initial values of nu
	 */
	private ArrayRealVector setNuInitial(final ArrayRealVector y) {	
		//nu.initial = expression( nu <- rep(4, length(y)))
		tempV = new ArrayRealVector(y.getDimension());
		tempV.set(4.0);
		return tempV;
	}

/** Calculates a first derivative of the likelihood 
	 * function in respect to supplied distribution parameter.
	 * @param whichDistParameter - distribution parameter
	 * @param y - vector of values of likelihood function
	 * @return vector of first derivative of the likelihood 
	 */
	public final ArrayRealVector firstDerivative(final int whichDistParameter, 
										   	     final ArrayRealVector  y) {
		setInterimArrays(y);
		tempV = null;		
		switch (whichDistParameter) {
	    case DistributionSettings.MU:
	    	tempV = dldm(y);
	       break;
	    case DistributionSettings.SIGMA:
	    	tempV = dlds(y);
	      break;
	    case DistributionSettings.NU:
	    	tempV = dldn(y);
	      break;
		default: 
			System.err.println("Requested first order "
					+ "derivative does not exist");
			break;
	    }			
		return tempV;
	}
	
	/** Set sigma1 array.
	 * @param y - response variable
	 */
	private void setInterimArrays(final ArrayRealVector y) {
		muV     = distributionParameters.get(DistributionSettings.MU);
		sigmaV  = distributionParameters.get(DistributionSettings.SIGMA);
	 	nuV     = distributionParameters.get(DistributionSettings.NU);
	 	
	 	size = y.getDimension();
		double[] temp = new double[size];
	 	for (int i = 0; i < size; i++) {
	 		
	 		//sigma1 <- (sqrt((nu-2)/nu))*sigma 
	 		temp[i] = (FastMath.sqrt((nuV.getEntry(i) - 2.0) 
	 						/ nuV.getEntry(i))) * sigmaV.getEntry(i);
	 	}
	 	sigma1 = new ArrayRealVector(temp, false);
	}
		
	/**  First derivative dldm = dl/dmu, where l - log-likelihood function.
	 * @param y - vector of values of response variable
	 * @return  a vector of first derivative dldm = dl/dmu
	 */	
	public final ArrayRealVector dldm(final ArrayRealVector y) { 
		
		//dldm <- TF()$dldm(y, mu, sigma1, nu)
		tf.setDistributionParameter(DistributionSettings.MU, muV);
		tf.setDistributionParameter(DistributionSettings.SIGMA, sigma1);
		tf.setDistributionParameter(DistributionSettings.NU, nuV);
		return tf.firstDerivative(DistributionSettings.MU, y);
	}

	/** First derivative dlds = dl/dsigma, where l - log-likelihood function.
	 * @param y - vector of values of response variable
	 * @return  a vector of First derivative dlds = dl/dsigma
	 */	
	public final ArrayRealVector dlds(final ArrayRealVector y) {
		
		tf.setDistributionParameter(DistributionSettings.MU, muV);
		tf.setDistributionParameter(DistributionSettings.SIGMA, sigma1);
		tf.setDistributionParameter(DistributionSettings.NU, nuV);
		tempV = tf.firstDerivative(DistributionSettings.SIGMA, y);
		
		double[] dlds = new double[size];
		ds1dd = new double[size];
		for (int i = 0; i < size; i++) {

			//ds1dd <-  (sqrt((nu-2)/nu))
			ds1dd[i] = FastMath.sqrt((nuV.getEntry(i) - 2) / nuV.getEntry(i));
			
			//dldd <- ds1dd*TF()$dldd(y, mu, sigma1, nu)
			dlds[i] =  ds1dd[i] * tempV.getEntry(i);
		}	
		return new ArrayRealVector(dlds, false);
	}

	/** First derivative dldn = dl/dnu, where l - log-likelihood function.
	 * @param y - vector of values of response variable
	 * @return  a vector of First derivative dldn = dl/dnu
	 */	
	public final ArrayRealVector dldn(final ArrayRealVector y) {
		
		tf.setDistributionParameter(DistributionSettings.MU, muV);
		tf.setDistributionParameter(DistributionSettings.SIGMA, sigma1);
		tf.setDistributionParameter(DistributionSettings.NU, nuV);
		
		tempV = tf.firstDerivative(DistributionSettings.SIGMA, y);
		tempV2 = tf.firstDerivative(DistributionSettings.NU, y);
		
		double[] dldn = new double[size];
		ds1dv = new double[size];
		for (int i = 0; i < size; i++) {

			//ds1dv <-  (sqrt(nu/(nu-2)))*sigma/(nu^2)
			ds1dv[i] = FastMath.sqrt(nuV.getEntry(i) / (nuV.getEntry(i) - 2))
					* (sigmaV.getEntry(i) / (nuV.getEntry(i) * nuV.getEntry(i)));
			
			//dldv <- TF()$dldv(y, mu, sigma1, nu) +
            //ds1dv*TF()$dldd(y, mu, sigma1, nu)
			dldn[i] = tempV2.getEntry(i) + ds1dv[i] * tempV.getEntry(i);
		}	
		return new ArrayRealVector(dldn, false);
		}

	/** Calculates a second derivative of the likelihood 
	 * function in respect to supplied distribution parameter.
	 * @param whichDistParameter - distribution parameter
	 * @param y - vector of values of likelihood function 
	 * @return vector of second derivative of the likelihood
	 */
	public final ArrayRealVector secondDerivative(final int whichDistParameter,
												  final ArrayRealVector y) {
				tempV = null;		
				switch (whichDistParameter) {
			      case DistributionSettings.MU:
			    	   tempV = d2ldm2(y);
			          break;
			      case DistributionSettings.SIGMA:
			    	 tempV = d2lds2(y);
			    	 break;
			      case DistributionSettings.NU:
			    	 tempV = d2ldn2(y);
			    	 break;
			      default: 
					System.err.println("Requested second order "
							+ "derivative does not exist");
					break;
			    }
				return tempV;
	}
		
	/** Second derivative d2ldm2= (d^2l)/(dmu^2),
	 *  where l - log-likelihood function.
	 * @param y - vector of values of response variable
	 * @return  a vector of second derivative d2ldm2= (d^2l)/(dmu^2)
	 */
	private ArrayRealVector d2ldm2(final ArrayRealVector y) {	

		double[] out = new double[size];
	 	for (int i = 0; i < size; i++) {

			//d2ldm2 <- -(nu+1)/((nu+3)*(sigma1^2))
	 		out[i] = -(nuV.getEntry(i) + 1) / ((nuV.getEntry(i) + 3) 
	 						* (sigma1.getEntry(i) * sigma1.getEntry(i)));
		}
	 	sigma1 = null;
	 	muV     = null;
	 	sigmaV  = null;
	 	nuV     = null;
		return new ArrayRealVector(out, false);
	}

	/** Second derivative d2lds2= (d^2l)/(dsigma^2), 
	 * where l - log-likelihood function.
	 * @param y - vector of values of response variable
	 * @return  a vector of second derivative d2lds2= (d^2l)/(dsigma^2)
	 */
	private ArrayRealVector d2lds2(final ArrayRealVector y) {
	
		double[] out = new double[size];
	 	for (int i = 0; i < size; i++) {
	 					
			//s2 <- sigma1^2
			final double s2 = sigma1.getEntry(i) * sigma1.getEntry(i);
			
			//d2ldd2 <- -(ds1dd^2)*((2*nu)/((nu+3)*s2))
			out[i] = -(ds1dd[i] * ds1dd[i]) * ((2.0 * nuV.getEntry(i))
											/ ((nuV.getEntry(i) + 3) * s2));
		}
	 	sigma1 = null;
	 	ds1dd   = null;
		muV 	= null;
		sigmaV 	= null;
		nuV 	= null;
		return new ArrayRealVector(out, false);
		
	}


	/** Second derivative d2ldn2= (d^2l)/(dnu^2), 
	 * where l - log-likelihood function.
	 * @param y - vector of values of response variable
	 * @return  a vector of second derivative d2ldn2= (d^2l)/(dnu^2)
	 */
	private ArrayRealVector d2ldn2(final ArrayRealVector y) {
		
	 	//d2ldv2 <- d2ldv2 + (ds1dv^2)*TF()$d2ldd2(sigma1, nu)
		tf.setDistributionParameter(DistributionSettings.MU, muV);
		tf.setDistributionParameter(DistributionSettings.SIGMA, sigma1);
		tf.setDistributionParameter(DistributionSettings.NU, nuV);
	 	
	 	tf.firstDerivative(DistributionSettings.SIGMA, y);
	 	tempV = tf.secondDerivative(DistributionSettings.SIGMA, y);

		double[] out = new double[size];
	 	for (int i = 0; i < size; i++) {
	 		
	        //v2 <- nu/2
			//v3 <-(nu+1)/2
	        //d2ldv2 <- trigamma(v3)-trigamma(v2)+(2*(nu+5))/(nu*(nu+1)*(nu+3))
			//d2ldv2 <- d2ldv2/4
	 		out[i] = (Gamma.trigamma((nuV.getEntry(i) + 1) / 2.0) 
	 				- Gamma.trigamma(nuV.getEntry(i) / 2.0) + (2.0 
	 						* (nuV.getEntry(i) + 5)) / (nuV.getEntry(i) 
	 								* (nuV.getEntry(i) + 1) * (nuV.getEntry(i)
	 															  + 3))) / 4.0;
	 		out[i] = out[i] + (ds1dv[i] * ds1dv[i]) * tempV.getEntry(i);
		}
	 	sigma1 = null;
	 	ds1dv   = null;
		muV 	= null;
		sigmaV 	= null;
		nuV 	= null;
		return new ArrayRealVector(out, false);
	}
	
	/** Calculates a second cross derivative of the likelihood 
	 * function in respect to supplied distribution parameters.
	 * @param whichDistParameter1 - first distribution parameter
	 * @param whichDistParameter2 - second distribution parameter
	 * @param y - vector of values of likelihood function 
	 * @return  vector of second cross derivative of the likelihood
	 */
	public final ArrayRealVector secondCrossDerivative(
												 final int whichDistParameter1,
												 final int whichDistParameter2, 
												 final ArrayRealVector y) {
				tempV = null;		
				if (whichDistParameter1 == DistributionSettings.MU) {
					switch (whichDistParameter2) {
					case DistributionSettings.SIGMA:
						tempV = d2ldmds(y);                      
						break;
					case DistributionSettings.NU:
						tempV = d2ldmdn(y);
						break;
			          default: 
					  	System.err.println("Second derivative does not exist");
					  	return null;
					}
				}
				if (whichDistParameter1 == DistributionSettings.SIGMA) {
					switch (whichDistParameter2) {
					case DistributionSettings.NU:
						tempV = d2ldsdn(y);
						break;
			          default: 
					  	System.err.println("Second derivative does not exist");
					  	return null;
					}
				}
				return tempV;
		}
	
	/** Second cross derivative of likelihood function in 
	 * respect to mu and sigma (d2ldmdd = d2l/dmu*dsigma).
	 * @param y - vector of values of response variable
	 * @return  a vector of Second cross derivative
	 */
	private ArrayRealVector d2ldmds(final ArrayRealVector y) {
		  //d2ldmdd = function(y) rep(0,length(y)
		  return new ArrayRealVector(y.getDimension());
	}
	
	/** Second cross derivative of likelihood function
 	 * in respect to mu and nu (d2ldmdd = d2l/dmu*dnu).
	 * @param y - vector of values of response variable
	 * @return  a vector of Second cross derivative
	 */
	private ArrayRealVector d2ldmdn(final ArrayRealVector y) {
			//d2ldmdv = function(y)  rep(0,length(y)),
		  return new ArrayRealVector(y.getDimension());
 	}	
	

	/** Second cross derivative of likelihood function 
	 * in respect to sigma and nu (d2ldmdd = d2l/dsigma*dnu).
	 * @param y - vector of values of response variable
	 * @return  a vector of Second cross derivative
	 */
	private ArrayRealVector d2ldsdn(final ArrayRealVector y) {	  
		
		sigmaV  = distributionParameters.get(DistributionSettings.SIGMA);
		nuV     = distributionParameters.get(DistributionSettings.NU);
		
		size = y.getDimension();
		double[] sigma1T = new double[size];
		double[] ds1ddT	 = new double[size];
		double[] ds1dvT = new double[size];
		double[] d2ldddvT = new double[size];
		double[] out = new double[size];
		for (int i = 0; i < size; i++) {
			//sigma1 <- (sqrt((nu-2)/nu))*sigma
			sigma1T[i] = FastMath.sqrt((nuV.getEntry(i) - 2.0) 
								/ nuV.getEntry(i)) * sigmaV.getEntry(i);
	
			//ds1dd <-  (sqrt((nu-2)/nu)) 
			ds1ddT[i] = (FastMath.sqrt((nuV.getEntry(i) 
												- 2.0) / nuV.getEntry(i)));
			
			// ds1dv <-  (sqrt(nu/(nu-2)))*sigma/(nu^2) 
			ds1dvT[i] = (FastMath.sqrt(nuV.getEntry(i) 
					/ (nuV.getEntry(i) - 2))) * sigmaV.getEntry(i) 
										/ (nuV.getEntry(i) * nuV.getEntry(i));
			
	        //d2ldddv <-  ds1dd*2/(sigma1*(nu+3)*(nu+1)) 
			d2ldddvT[i] = ds1ddT[i] * 2 / (sigma1T[i] * (nuV.getEntry(i)
												+ 3) * (nuV.getEntry(i) + 1));
		}
		
		//d2ldddv <- d2ldddv + ds1dd*ds1dv*TF()$d2ldd2(sigma1, nu)
		tf.setDistributionParameter(DistributionSettings.SIGMA, 
										new ArrayRealVector(sigma1T, false));
		tf.setDistributionParameter(DistributionSettings.NU, nuV);
		
	 	tf.firstDerivative(DistributionSettings.SIGMA, y);
	 	tempV = tf.secondDerivative(DistributionSettings.SIGMA, y);
		
		for (int i = 0; i < size; i++) {
			out[i] = d2ldddvT[i] + ds1ddT[i] * ds1dvT[i] * tempV.getEntry(i);
		}
		tempV 	= null;
		sigmaV 	= null;
		nuV 	= null;
		return new ArrayRealVector(out, false);
	}
		
	/** Computes the global Deviance Increament.
	 * @param y - vector of response variable values
	 * @return vector of global Deviance Increament values 
	 */
	public final ArrayRealVector globalDevianceIncreament(
													final ArrayRealVector y) {
        //G.dev.incr  = function(y,mu,sigma,nu,tau,...)  
		size = y.getDimension();
		double[] out = new double[size];

		double[] muArr = distributionParameters.get(
									DistributionSettings.MU).getDataRef();
		double[] sigmaArr = distributionParameters.get(
									DistributionSettings.SIGMA).getDataRef();
		double[] nuArr = distributionParameters.get(
									DistributionSettings.NU).getDataRef();
		
		for (int i = 0; i < size; i++) {
			
			out[i] = (-2) * dTF2(y.getEntry(i), muArr[i], sigmaArr[i], 
								nuArr[i], Controls.LOG_LIKELIHOOD);
		}
		return new ArrayRealVector(out, false);
	}
	
	/** Computes the probability density function (PDF) of this 
	 * distribution evaluated at the specified point x.
	 * @param x - value of response variable
	 * @param mu - value of mu distribution parameter
	 * @param sigma - value of sigma distribution parameter
	 * @param nu - value of nu distribution parameter
	 * @param isLog  - logical, whether to take log of the function or not
	 * @return value of probability density function
	 */
	public final double dTF2(final double x, 
							 final double mu, 
							 final double sigma, 
							 final double nu, 
							 final boolean isLog) {
		
		// {  if (any(sigma <= 0))stop(paste("sigma must be positive",))
		if (sigma <= 0) {
			System.err.println("sigma must be positive");
			return -1.0;
		}
		
		//if (any(nu <= 0))  stop(paste("nu must be positive", "\n", ""))
		if (nu <= 0) {
			System.err.println("nu must be positive");
			return -1.0;
		}
		
		double out = 0;
		
		//sigma1 <- (sqrt((nu-2)/nu))*sigma
		final double sigma1 = FastMath.sqrt((nu - 2)/nu) * sigma;
		
		//ifelse(nu>1000000, dNO(x,mu=mu,sigma=sigma1,log=FALSE),
		//(1/sigma1)*dt((x-mu)/sigma1, df=nu, log =FALSE))
		if (nu > 1000000) {
			noDist.setDistrParameters(mu, sigma1);
			out = noDist.density(x);
		} else {
			tdDist.setDegreesOfFreedom(nu);
			out = (1 / sigma1) * tdDist.density((x - mu) / sigma1);
		}
		
		if (isLog) {
			out = FastMath.log(out);
		}
		return out;
	}

	/**
	 * dTF2(x) launches dTF2(x, mu, sigma, nu, isLog)
	 *  with deafult mu=0, sigma=1, nu=10, isLof=false.
	 * @param x - value of response variable
	 * @return value of probability density function 
	 */
	//dTF2<-function(x, mu=0, sigma=1, nu=10, log=FALSE)
	public final double dTF2(final double  x) {
		return dTF2(x, 0.0, 1.0, 10.0, false);
	}

		/** Computes the cumulative distribution 
		 * function P(X <= q) for a random variable X .
		 * whose values are distributed according to this distribution
		 * @param q - value of quantile
		 * @param mu - value of mu distribution parameter
		 * @param sigma - value of sigma distribution parameter
		 * @param nu - value of nu distribution parameter 
		 * @param lowerTail - logical, if TRUE (default), probabilities
		 *  are P[X <= x] otherwise, P[X > x].
		 * @param isLog - logical, if TRUE, probabilities p are given as log(p)
		 * @return value of cumulative probability function values P(X <= q)
		 */
		public final double pTF2(final double q, 
					final double mu, 
					final double sigma, 
					final double nu,
					final boolean lowerTail, 
					final boolean isLog) {
			
			// {  if (any(sigma <= 0))stop(paste("sigma must be positive",))
			if (sigma <= 0) {
				System.err.println("sigma must be positive");
				return -1.0;
			}
			
			//if (any(nu <= 0))  stop(paste("nu must be positive", "\n", ""))
			if (nu <= 0) {
				System.err.println("nu must be positive");
				return -1.0;
			}
		
			double out = 0;
			
			//sigma1 <- (sqrt((nu-2)/nu))*sigma
			final double sigma1 = FastMath.sqrt((nu - 2)/nu) * sigma;
			
			//cdf <- ifelse(nu>1000000, 
			//pNO(q, mu=mu, sigma=sigma1, 
			//lower.tail=lower.tail, log.p=log.p),
			//pt((q-mu)/sigma1, df=nu,   lower.tail=lower.tail, log.p=log.p))
			if (nu > 1000000) {
				
				noDist.setDistrParameters(mu, sigma1);
				out = noDist.cumulativeProbability(q);
				if (!lowerTail) {
				    	if (isLog) {
				    		
				    		out = FastMath.log(1 - out);
				    	} else {
				    		
				    		out = 1 - out;
				    	}
				    } else if (isLog) {
				    	
				    	out = FastMath.log(out);
				    } 
			} else {
				
				tdDist.setDegreesOfFreedom(nu);
				out = tdDist.cumulativeProbability((q - mu) / sigma1);
				if (!lowerTail) {
			    	if (isLog) {
			    		
			    		out = FastMath.log(1 - out);
			    	} else {
			    		
			    		out = 1 - out;
			    	}
			    } else if (isLog) {
			    	
			    	out = FastMath.log(out);
			    } 
			}
			return out;	
		}
	
	/**
   * pTF2(q) launches pTF2(q, mu, sigma, nu, isLog)
   *  with deafult mu=0, sigma=1, nu=10, lowerTail = TRUE, isLog=false.
	 * @param q - quantile
	 * @return value of cumulative probability function P(X <= q)
	 */
	//pTF2 <- function(q, mu=0, sigma=1, nu=10, 
	//lower.tail = TRUE, log.p = FALSE)
	public final double pTF2(final double x) {	
		return pTF2(x, 0.0, 1.0, 10.0, true, false);
	}
	

	/** Computes the quantile (inverse cumulative probability)
	 *  function  of this distribution.
	* @param p - value of cumulative probability
	* @param mu -  value of mu distribution parameters
	* @param sigma -  value of sigma distribution parameters
	* @param nu -  value of nu distribution parameters 
	* @param lowerTail - logical; if TRUE (default), probabilities 
	* are P[X <= x] otherwise, P[X > x]
	* @param isLog - logical; if TRUE, probabilities p are given as log(p).
	* @return value of quantile function
	*/
	public final double qTF2(final double p, 
							 final double mu, 
							 final double sigma, 
							 final double nu, 
							 final boolean lowerTail, 
							 final boolean isLog) {
		
		// {  if (any(sigma <= 0))stop(paste("sigma must be positive",))
		if (sigma <= 0) {
			System.err.println("sigma must be positive");
			return -1.0;
		}
		
		//if (any(nu <= 0))  stop(paste("nu must be positive", "\n", ""))
		if (nu <= 0) {
			System.err.println("nu must be positive");
			return -1.0;
		}
		
		double temp = p;
		if (isLog) {
			
			temp = FastMath.exp(temp);
		}
		
		if (temp < 0 || temp > 1) {
			
			System.err.println("Error: p must be between 0 and 1");
		}	
	    if (!lowerTail) {
	    	 temp = 1 - temp;
	    }

	    double out = 0;
	    
		//sigma1 <- (sqrt((nu-2)/nu))*sigma
		final double sigma1 = FastMath.sqrt((nu - 2) / nu) * sigma;
		
		 //ifelse(nu>1000000, qNO(p, mu=mu, sigma=sigma1, 
		 //lower.tail=lower.tail, log.p=log.p),
	     //mu+sigma1*qt(p,df=nu, lower.tail = lower.tail))
		if (nu > 1000000) {
			noDist.setDistrParameters(mu, sigma1);
			out = noDist.inverseCumulativeProbability(temp);
		} else {
			if (isLog) {
				temp = FastMath.exp(temp);
			}
			tdDist.setDegreesOfFreedom(nu);
			out = mu + sigma1 * tdDist.inverseCumulativeProbability(temp);
		}
		return out;	
	}
	
	/**
	* qTF2(p) launches qTF2(p, mu, sigma, nu, isLog)
	*  with deafult mu=0, sigma=1, nu=10,.
	* lowerTail = true, isLof=false
	* @param p - value of cumulative probability
	* @return quantile
	*/
	//qTF2 <- function(p, mu=0, sigma=1, nu=10,
	//lower.tail = TRUE, log.p = FALSE)
	private final double qTF2(final double p){
		return pTF2(p, 0.0, 1.0, 10.0, true, false);
	}

	/** Generates a random sample from this distribution.
	 * @param mu -  vector of mu distribution parameters values
	 * @param sigma -  vector of sigma distribution parameters values
	 * @param nu -  vector of nu distribution parameters values
	 * @param uDist -  object of UniformRealDistribution class;
	 * @return random sample vector
	 */	
		public final double rTF2(final double mu, 
			 		 			 final double sigma, 
			 		 			 final double nu, 
			 		 			 final UniformRealDistribution uDist) {
			
			// {  if (any(sigma <= 0))stop(paste("sigma must be positive",))
			if (sigma <= 0) {
				System.err.println("sigma must be positive");
				return -1.0;
			}
			
			//if (any(nu <= 0))  stop(paste("nu must be positive", "\n", ""))
			if (nu <= 0) {
				System.err.println("nu must be positive");
				return -1.0;
			}		    
		//r <- qST3(p,mu=mu,sigma=sigma,nu=nu,tau=tau)
		return qTF2(uDist.sample(), mu, sigma, nu, true, false);
	}
	
	/**
	* rTF2(uDist) launches rTF2(mu, sigma, nu, uDist) with deafult
	*  mu=0, sigma=1, nu=10.
		* @param uDist -  object of UniformRealDistribution class;
	 * @return random sample value
	*/
	//rTF2 <- function(n, mu=0, sigma=1, nu=10)
	private final double rTF2( final UniformRealDistribution uDist) {
		return rTF2(0.0, 1.0, 10.0, uDist);
	}
	
	/**
	* Checks whether the mu distribution parameter is valid.
	* @param y - vector of response variavbles
	* @return - boolean
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
		boolean tempB = false;
		switch (whichDistParameter) {
      case DistributionSettings.MU:
      	tempB = isMuValid(
      				distributionParameters.get(DistributionSettings.MU));
         break;
      case DistributionSettings.SIGMA:
      	tempB = isSigmaValid(
      				distributionParameters.get(DistributionSettings.SIGMA));
        break;
      case DistributionSettings.NU:
      	tempB = isNuValid(
      				distributionParameters.get(DistributionSettings.NU));
        break;
		default: System.out.println("The specific distribution parameter"
				+ " does not exist for this distribution");
		}
		return tempB;
	}

	/**
	* Checks whether the mu distribution parameter is valid.
	* @param mu - vector of mu (mean) values
	* @return - boolean
	*/
	private boolean isMuValid(final ArrayRealVector mu) {
		//mu.valid = function(mu) TRUE,
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

	/**
	 * Checks whether the nu distribution parameter is valid.
	 * @param nu - vector of nu values
	 * @return - - boolean
	 */
	private boolean isNuValid(final ArrayRealVector nu) {
		return nu.getMinValue() > 0;	
	}



	/**
	 * Get number of distribution parameters.
	 * @return number of distribution parameters
	 */
	public final int getNumberOfDistribtionParameters() {
		return numDistPar;
	}


	/**
	 * Get type of distributuion (Continuous, Discrete or Mixed).
	 * @return type of distributuion
	 */
	public final int getTypeOfDistribution() {
		return DistributionSettings.CONTINUOUS;
	}


	/**
	 * Get distribution name.
	 * @return distribution name.
	 */
	public final int getFamilyOfDistribution() {
		return DistributionSettings.TF2;
	}


	/**
	 * Set distribution parsameters.
	 * @param whichDistParameter - the fitting distribution parameter
	 * @param fvDistributionParameter - vector of values of 
	 * fitting distribution parameter
	 */
	public final void setDistributionParameter(
							final int whichDistParameter, 
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