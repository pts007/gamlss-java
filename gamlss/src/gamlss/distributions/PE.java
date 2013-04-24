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

import org.apache.commons.math3.distribution.GammaDistribution;
import org.apache.commons.math3.distribution.TDistribution;
import org.apache.commons.math3.distribution.UniformRealDistribution;
import org.apache.commons.math3.exception.NumberIsTooLargeException;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.special.Gamma;
import org.apache.commons.math3.stat.descriptive.moment.Mean;
import org.apache.commons.math3.stat.descriptive.moment.StandardDeviation;
import org.apache.commons.math3.util.FastMath;

/**
 * @author Dr. Vlasios Voudouris, Daniil Kiose, 
 * Prof. Mikis Stasinopoulos and Dr Robert Rigby.
 */
public class PE implements GAMLSSFamilyDistribution {
	
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
	/** Array of first derrivative values dl/dmu. */
	private double[] dldm;
	/** Temporary vector for interim operations. */
	private ArrayRealVector tempV;	
	/** Temporary array for interim operations. */
	private double[] logC;
	/** Temporary array for interim operations. */
	private double[] c;
	/** Temporary array for interim operations. */
	private double[] z;
	/** Temporary int for interim operations. */
	private int size;
	/** Object of the Gamma distribution .*/
	private GammaDistr gammaDistr;
	
	/** This is the Power Exponential distribution  with default link
	 *  (muLink="identity",sigmaLink="log", nuLink="log"). */
	public PE() {
	
		this(DistributionSettings.IDENTITY, 
			 DistributionSettings.LOG, 
			 DistributionSettings.LOG);
	}
	
	/** This is the Power Exponential distribution with supplied link 
	 * function for each of the distribution parameters. 
	 * @param muLink - link function for mu distribution parameter
	 * @param sigmaLink - link function for sigma distribution parameter
	 * @param nuLink - link function for nu distribution parameter*/
	public PE(final int muLink, 
			  final int sigmaLink, 
			  final int nuLink) {
	
		distributionParameterLink.put(DistributionSettings.MU, 
				MakeLinkFunction.checkLink(DistributionSettings.PE, muLink));
		distributionParameterLink.put(DistributionSettings.SIGMA,  
				MakeLinkFunction.checkLink(DistributionSettings.PE, sigmaLink));
		distributionParameterLink.put(DistributionSettings.NU,  
				MakeLinkFunction.checkLink(DistributionSettings.PE, nuLink));
		
		gammaDistr = new GammaDistr();
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
		//sigma.initial = expression( sigma <- (abs(y-mean(y))+sd(y))/2 )	
		final double mean = new Mean().evaluate(y.getDataRef());
		final double sd = new StandardDeviation().evaluate(y.getDataRef());	
		size = y.getDimension();
		double[] out = new double[size];
		for (int i = 0; i < size; i++) {
			out[i] = (FastMath.abs(y.getEntry(i) - mean) + sd) / 2;
		}
		return new ArrayRealVector(out, false);
	}

	/** Calculate and set initial value of nu.
	 * @param y - vector of values of response variable
	 * @return vector of initial values of nu
	 */
	private ArrayRealVector setNuInitial(final ArrayRealVector y) {	
		//nu.initial = expression( nu <- rep(1.8, length(y)))
		tempV = new ArrayRealVector(y.getDimension());
		tempV.set(1.8);
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
	
	/** Set logC, c, z arrays.
	 * @param y - response variable
	 */
	private void setInterimArrays(final ArrayRealVector y) {
		muV     = distributionParameters.get(DistributionSettings.MU);
		sigmaV  = distributionParameters.get(DistributionSettings.SIGMA);
	 	nuV     = distributionParameters.get(DistributionSettings.NU);
		
	 	size = y.getDimension();
		logC = new double[size];
		c = new double[size];
		z = new double[size];
	 	for (int i = 0; i < size; i++) {
	 		
			//log.c <- 0.5*(-(2/nu)*log(2)+lgamma(1/nu)-lgamma(3/nu))
			logC[i] =  0.5 * (-(2 / nuV.getEntry(i)) * FastMath.log(2) 
					+ Gamma.logGamma(1 / nuV.getEntry(i)) - Gamma.logGamma(3 
															/nuV.getEntry(i)));
			
		 	//c <- exp(log.c)
	 	    c[i] = FastMath.exp(logC[i]);
		
	 	    //z <- (y-mu)/sigma
	 	    z[i] = (y.getEntry(i) - muV.getEntry(i)) / sigmaV.getEntry(i);
	 	}
	}

	/**  First derivative dldm = dl/dmu, where l - log-likelihood function.
	 * @param y - vector of values of response variable
	 * @return  a vector of first derivative dldm = dl/dmu
	 */	
	public final ArrayRealVector dldm(final ArrayRealVector y) { 

	 	dldm = new double[size];
	 	for (int i = 0; i < size; i++) {			
		 	 
	 		//dldm <- (sign(z)*nu)/(2*sigma*abs(z));
	 		dldm[i] = (FastMath.signum(z[i]) * nuV.getEntry(i)) 
	 				/ (2 * sigmaV.getEntry(i) * FastMath.abs(z[i]));
		 	    
		 	//dldm <- dldm*((abs(z/c))^nu) 
	 		dldm[i] = dldm[i] * (FastMath.pow(FastMath.abs(z[i] 
	 									/ c[i]), nuV.getEntry(i)));
		}	
		logC = null;
		c = null;
		z = null;
	  	return new ArrayRealVector(dldm, false);
		}

	/** First derivative dlds = dl/dsigma, where l - log-likelihood function.
	 * @param y - vector of values of response variable
	 * @return  a vector of First derivative dlds = dl/dsigma
	 */	
	public final ArrayRealVector dlds(final ArrayRealVector y) {	
		
		double[] dlds = new double[size];
	 	for (int i = 0; i < size; i++) {
	 		
	 	    //dldd <- ((nu/2)*((abs(z/c))^nu)-1)/sigma
	 		dlds[i] = ((nuV.getEntry(i) / 2) 
	 				* (FastMath.pow((FastMath.abs(z[i] / c[i])), 
	 						nuV.getEntry(i))) - 1) / sigmaV.getEntry(i);
		}	
		logC = null;
		c = null;
		z = null;
		return new ArrayRealVector(dlds, false);
	}

	/** First derivative dldn = dl/dnu, where l - log-likelihood function.
	 * @param y - vector of values of response variable
	 * @return  a vector of First derivative dldn = dl/dnu
	 */	
	public final ArrayRealVector dldn(final ArrayRealVector y) {	

		double[] dldn = new double[size];
	 	for (int i = 0; i < size; i++) {
			
		 	//dlogc.dv <- (1/(2*nu^2))*(2*log(2)-digamma(1/nu)+3*digamma(3/nu))
		 	final double dlogcDv = (1 / (2 * nuV.getEntry(i) 
		 			* nuV.getEntry(i))) * (2 * FastMath.log(2) 
		 					- Gamma.digamma(1 / nuV.getEntry(i)) 
		 					+ 3 * Gamma.digamma(3 / nuV.getEntry(i)));
			
		 	//dldv <- (1/nu)-0.5*((log(abs(z/c)))*((abs(z/c))^nu)) 
		 	dldn[i] = (1 / nuV.getEntry(i)) - 0.5  
		 			* ((FastMath.log(FastMath.abs(z[i] / c[i]))) 
		 					* (FastMath.pow(FastMath.abs(z[i] / c[i]), 
		 												nuV.getEntry(i))));
		 	    
		 	    
		 	//dldv <- dldv+log(2)/(nu^2)+digamma(1/nu)/(nu^2)
		 	dldn[i] = dldn[i] + FastMath.log(2) / (nuV.getEntry(i) 
		 			* nuV.getEntry(i)) + Gamma.digamma(1 / nuV.getEntry(i)) 
		 								/ (nuV.getEntry(i) * nuV.getEntry(i));
		 	
		 	    //dldv <- dldv+(-1+(nu/2)*((abs(z/c))^nu))*dlogc.dv
		 	dldn[i] = dldn[i] + (-1 + (nuV.getEntry(i) / 2) 
		 					* (FastMath.pow(FastMath.abs(z[i] / c[i]), 
		 										nuV.getEntry(i)))) * dlogcDv;
		}	
		logC = null;
		c = null;
		z = null;
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
		 	    
		    //d2ldm2 <- if (any(nu<1.05)) -dldm*dldm else d2ldm2
	 		if (nuV.getEntry(i) < 1.05) {
	 			
	 			out[i] = -dldm[i] * dldm[i];
		 	} else {
		 		
		 		//d2ldm2 <- -(nu*nu*gamma(2-(1/nu))*gamma(3/nu))
		 		///((sigma*gamma(1/nu))^2)
		 		out[i] = -(nuV.getEntry(i) * nuV.getEntry(i) 
		 				* FastMath.exp(Gamma.logGamma(2 
		 		       - (1 / nuV.getEntry(i)))) * FastMath.exp(
		 				Gamma.logGamma(3 / nuV.getEntry(i)))) 
		 					/ ((sigmaV.getEntry(i) * FastMath.exp(
		 						Gamma.logGamma(1 / nuV.getEntry(i)))) 
		 							* (sigmaV.getEntry(i) * FastMath.exp(
		 												    Gamma.logGamma(1 
		 												 / nuV.getEntry(i)))));
		 	}
		}	
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
	 		
	 	    //d2ldd2 <- -nu/(sigma^2)
	 	    out[i] = -nuV.getEntry(i) / (sigmaV.getEntry(i) 
	 	    									* sigmaV.getEntry(i));
		}	
	 	muV     = null;
	 	sigmaV  = null;
	 	nuV     = null;
		return new ArrayRealVector(out, false);
	}
	
	/** Second derivative d2ldn2= (d^2l)/(dnu^2), 
	 * where l - log-likelihood function.
	 * @param y - vector of values of response variable
	 * @return  a vector of second derivative d2ldn2= (d^2l)/(dnu^2)
	 */
	private ArrayRealVector d2ldn2(final ArrayRealVector y) {	

		double[] out = new double[size];
	 	for (int i = 0; i < size; i++) {
	 		
	 	    //dlogc.dv <- (1/(2*nu^2))*(2*log(2)-digamma(1/nu)+3*digamma(3/nu))
	 	    final double dlogcDv = (1 / (2 * nuV.getEntry(i) * 
	 	    		nuV.getEntry(i))) * (2 * FastMath.log(2) - 
	 	    				Gamma.digamma(1 / nuV.getEntry(i)) 
	 	    				+ 3 * Gamma.digamma(3 / nuV.getEntry(i)));
	 	    
	 	    //p <- (1+nu)/nu 
	 	    final double p = (1 + nuV.getEntry(i)) / nuV.getEntry(i);
	 	    
            //part1 <- p*trigamma(p)+2*(digamma(p))^2
	 	    final double part1 = p * Gamma.trigamma(p) 
	 	    		+ 2 * FastMath.pow(Gamma.digamma(p), 2);
	 	    
	 	    //part2 <- digamma(p)*(log(2)+3-3*digamma(3/nu)-nu)
	 	    final double part2 = Gamma.digamma(p) * (FastMath.log(2) 
	 			   + 3 - 3 * Gamma.digamma(3 / nuV.getEntry(i)) 
	 			   									- nuV.getEntry(i));
	 	    
	 	    //part3 <- -3*(digamma(3/nu))*(1+log(2))    
	 	    final double part3 = -3 * (Gamma.digamma(3 / nuV.getEntry(i)))
	 	    										* (1 + FastMath.log(2));
	 	    
	 	    //part4 <- -(nu+log(2))*log(2)
	 	    final double part4 = -(nuV.getEntry(i) + FastMath.log(2)) 
	 			   										* FastMath.log(2);
	 	    
	 	    //part5 <- -nu+(nu^4)*(dlogc.dv)^2
	 	   	final double part5 = -nuV.getEntry(i) 
	 	   			+ FastMath.pow(nuV.getEntry(i), 4) * dlogcDv * dlogcDv;
	 	    
	 	    //d2ldv2 <- part1+part2+part3+part4+part5
	 	    out[i] = part1+part2+part3+part4+part5;
	 	    
	 	    //d2ldv2 <- -d2ldv2/nu^3  
	 	    out[i] = -out[i]/(FastMath.pow(nuV.getEntry(i),3));
	 	   
	 	    //d2ldv2 <- ifelse(d2ldv2 < -1e-15, d2ldv2,-1e-15)
	 	   if (out[i] > -1e-15) {
	 		  out[i] = -1e-15;
	 	   }
	 	}	
	 	muV     = null;
	 	sigmaV  = null;
	 	nuV     = null;
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
		  // d2ldmdd = function(y)  rep(0,length(y))
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
		
		ArrayRealVector sigmaT  
		= distributionParameters.get(DistributionSettings.SIGMA);
		ArrayRealVector nuT    
		= distributionParameters.get(DistributionSettings.NU);
		
		double[] out = new double[size];
	 	for (int i = 0; i < size; i++) {
			
	 		//d2ldddv = function(y,mu,sigma,nu) (1/(2*sigma))
	 		//*((3/nu)*(digamma(1/nu)-digamma(3/nu))+2+(2/nu))
			out[i] = (1 / (2 * sigmaT.getEntry(i))) 
					* ((3 / nuT.getEntry(i)) * (Gamma.digamma(1 
							/ nuT.getEntry(i)) - Gamma.digamma(3 
									/ nuT.getEntry(i))) + 2 
											+ (2 / nuT.getEntry(i)));
		}
		return new ArrayRealVector(out, false);
	}		
		
	/** Computes the global Deviance Increament.
	 * @param y - vector of response variable values
	 * @return vector of global Deviance Increament values 
	 */
	public final ArrayRealVector globalDevianceIncreament(
													final ArrayRealVector y) {
        //G.dev.incr  = function(y,mu,sigma,nu,tau,...)  
		//-2*dST3(y,mu,sigma,nu,tau,log=TRUE),
		size = y.getDimension();
		double[] out = new double[size];

		double[] muArr = distributionParameters.get(
									DistributionSettings.MU).getDataRef();
		double[] sigmaArr = distributionParameters.get(
									DistributionSettings.SIGMA).getDataRef();
		double[] nuArr = distributionParameters.get(
									DistributionSettings.NU).getDataRef();
		
		for (int i = 0; i < size; i++) {
			
			out[i] = (-2) * dPE(y.getEntry(i), muArr[i], sigmaArr[i], 
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
		public final double dPE(final double x, 
				 				final double mu, 
				 				final double sigma, 
				 				final double nu, 
				 				final boolean isLog) {
			
			// {  if (any(sigma < 0))stop(paste("sigma must be positive",))
			if (sigma < 0) {
				System.err.println("sigma must be positive");
				return -1.0;
			}
			
			//if (any(nu < 0))  stop(paste("nu must be positive", "\n", ""))
			if (nu < 0) {
				System.err.println("nu must be positive");
				return -1.0;
			}
		
			//log.c <- 0.5*(-(2/nu)*log(2)+lgamma(1/nu)-lgamma(3/nu))
			final double logC =  0.5 * (-(2 / nu) * FastMath.log(2) 
					+ Gamma.logGamma(1 / nu) - Gamma.logGamma(3 / nu));
			
		 	//c <- exp(log.c)
			final double c = FastMath.exp(logC);
		
	 	    //z <- (y-mu)/sigma
			final double z = (x - mu) / sigma;
			  
			
	 	    //log.lik <- -log(sigma)+log(nu)-log.c-(0.5
			//*(abs(z/c)^nu))-(1+(1/nu))*log(2)-lgamma(1/nu)
			double out = -FastMath.log(sigma) + FastMath.log(nu) 
					- logC - (0.5 * (FastMath.pow(FastMath.abs(z / c), 
							nu))) - (1 + (1 / nu)) * FastMath.log(2) 
												- Gamma.logGamma(1 / nu);
			
			//if(log==FALSE) fy  <- exp(log.lik) else fy <- log.lik
			if (!isLog) {
				out = FastMath.exp(out);
			}
			return out;
		}
		
		/** dPE(x) launches dPE(x, mu, sigma, nu, isLog) with 
		 *deafult mu=0, sigma=1, nu=2, log=FALSE.
		 * @param x - vector of response variable values
		 * @return vector of probability density function values
		 */
		//dPE<-function(x, mu=0, sigma=1, nu=2, log=FALSE)
		public final double dPE(final double x) {
			return dPE(x, 0.0, 1.0, 2.0, false);
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
		public final double pPE(final double q, 
     	   		 				final double mu, 
     	   		 				final double sigma, 
     	   		 				final double nu,
     	   		 				final boolean lowerTail, 
     	   		 				final boolean isLog) {
			
		// {  if (any(sigma < 0))stop(paste("sigma must be positive",))
		if (sigma < 0) {
			System.err.println("sigma must be positive");
			return -1.0;
		}
		
		//if (any(nu < 0))  stop(paste("nu must be positive", "\n", ""))
		if (nu < 0) {
			System.err.println("nu must be positive");
			return -1.0;
		}
		
		double out = 0;
		  
 	    //ifelse(nu>10000, (q-(mu-sqrt(3)*sigma))/(sqrt(12)*sigma),cdf)
		if(nu > 10000) {
			out = (q - (mu - FastMath.sqrt(3) * sigma)) 
										/ (FastMath.sqrt(12) * sigma);
		} else {
			
			//log.c <- 0.5*(-(2/nu)*log(2)+lgamma(1/nu)-lgamma(3/nu))
			final double logC =  0.5 * (-(2 / nu) * FastMath.log(2) 
					+ Gamma.logGamma(1 / nu) - Gamma.logGamma(3 / nu));
			
		 	//c <- exp(log.c)
			final double c = FastMath.exp(logC);
		
	 	    //z <- (y-mu)/sigma
			final double z = (q - mu) / sigma;

			//s <- 0.5*((abs(z/c))^nu)
			final double s = 0.5 * (FastMath.pow(FastMath.abs(z / c), nu));
	 	    
	 	    //cdf <- 0.5*(1+pgamma(s,shape=1/nu,scale=1)*sign(z))
			out = 0.5 * (1 + Gamma.regularizedGammaP(1 
											/ nu, s) * FastMath.signum(z));
		}
 	    
		//if(lower.tail==TRUE) cdf  <- cdf else  cdf <- 1-cdf 
 	    //if(log.p==FALSE) cdf  <- cdf else  cdf <- log(cdf) 
		if (!lowerTail) {
	    	if (isLog) {
	    		out = FastMath.log(1 - out);
	    	} else {
	    		out = 1 - out;
	    	}
	    } else if (isLog) {
	    	//if(log.p==FALSE) cdf  <- cdf else  cdf <- log(cdf)
	    	out = FastMath.log(out);
	    }
		return out;
	}

	/**
	 * pPE(q) launches pPE(q, mu, sigma, nu, isLog) 
	 * with deafult mu=0, sigma=1, nu=2, lowerTail = TRUE, isLog=false.
	 * @param q - quantile
	 * @return value of cumulative probability function P(X <= q)
	 */
	//pPE<- function(q, mu=0, sigma=1, nu=2, lower.tail = TRUE, log.p = FALSE)
	public final double pPE(final double q) {
		return pPE(q, 0.0, 1.0, 2.0, true, false);
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
	public final double qPE(final double p, 
	         				final double mu, 
	         				final double sigma, 
	         				final double nu, 
	         				final boolean lowerTail, 
	         				final boolean isLog) {
		
		// {  if (any(sigma < 0))stop(paste("sigma must be positive",))
		if (sigma < 0) {
			System.err.println("sigma must be positive");
			return -1.0;
		}
		
		//if (any(nu < 0))  stop(paste("nu must be positive", "\n", ""))
		if (nu < 0) {
			System.err.println("nu must be positive");
			return -1.0;
		}
		
		//if (log.p==TRUE) p <- exp(p) else p <- p
		double out  = p;
		if(isLog) {
			
			//if (lower.tail==TRUE) p <- p else p <- 1-p
			if(!lowerTail) {
				
				out = 1 - FastMath.exp(out);
				//out = FastMath.exp(1 -out);
			} else {
				
				out = FastMath.exp(out);
			}
		} else if (!lowerTail) {
			
			out = 1 - out;
		}
	
		//if (any(p < 0)|any(p > 1))  
		//stop(paste("p must be between 0 and 1", "\n", "")
		if (out < 0 || out > 1) {
			System.err.println("p must be between 0 and 1");
			return -1.0;
		}
		
		//log.c <- 0.5*(-(2/nu)*log(2)+lgamma(1/nu)-lgamma(3/nu))
		final double logC =  0.5 * (-(2 / nu) * FastMath.log(2) 
				+ Gamma.logGamma(1 / nu) - Gamma.logGamma(3 / nu));
		
	 	//c <- exp(log.c)
		final double c = FastMath.exp(logC);
	
	    //suppressWarnings(
		//s <- qgamma((2*p-1)*sign(p-0.5),shape=(1/nu),scale=1))
		gammaDistr.setDistrParameters(1 / nu, 1.0);
		final double s = gammaDistr.inverseCumulativeProbability((2 * out - 1)
												* FastMath.signum(out - 0.5));
		    
		//z <- sign(p-0.5)*((2*s)^(1/nu))*c
		final double z = FastMath.signum(out - 0.5) 
									* (FastMath.pow((2 * s), (1 / nu))) * c;
		    
		//ya <- mu + sigma*z 
		 out = mu + sigma * z;

		 return out;
	}

	/**
	* qPE(p) launches qPE(p, mu, sigma, nu, isLog)
	*  with deafult mu=0, sigma=1, nu=2.
	* lowerTail = true, isLof=false
	* @param p - value of cumulative probability 
	* @return quantile
	*/
	//qPE<- function(p, mu=0, sigma=1, nu=2, lower.tail = TRUE, log.p = FALSE)
	private final double qPE(final double p){
		return qPE(p, 0.0, 1.0, 2.0, true, false);
	}
	
	/** Generates a random sample from this distribution.
	 * @param mu -  vector of mu distribution parameters values
	 * @param sigma -  vector of sigma distribution parameters values
	 * @param nu -  vector of nu distribution parameters values
	 * @param uDist -  object of UniformRealDistribution class;
	 * @return random sample vector
	 */	
		public final double  rPE(final double mu, 
		       	   				 final double sigma, 
		       	   				 final double nu, 
		       	   				 final UniformRealDistribution uDist) {
			
		// {if (any(sigma <= 0))stop(paste("sigma must be positive"))
		if (sigma <= 0) {
			System.err.println("sigma must be positive");
			return -1.0;
		}
		//if (any(nu <= 0))  stop(paste("nu must be positive", "\n", ""))
		if (nu <= 0) {
			System.err.println("nu must be positive");
			return -1.0;
		}
		
		//r <- qPE(p,mu=mu,sigma=sigma,nu=nu)
		return qPE(uDist.sample(), mu, sigma, nu, true, false);
	}
	
	/**
	* rPE(n) launches rPE(n, mu, sigma, nu)
	*  with deafult mu=0, sigma=1, nu=2.
	 * @param uDist -  object of UniformRealDistribution class;
	 * @return random sample value
	*/
	//rPE <- function(n, mu=0, sigma=1, nu=2)
	private final double rPE(final UniformRealDistribution uDist){
		return rPE(0.0, 1.0, 2.0, uDist);
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
		return DistributionSettings.PE;
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

