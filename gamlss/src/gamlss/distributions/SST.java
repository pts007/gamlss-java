/*
  Copyright 2012 by Dr. Vlasios Voudouris and ABM Analytics Ltd
  Licensed under the Academic Free License version 3.0
  See the file "LICENSE" for more information
*/
package gamlss.distributions;

import gamlss.utilities.Controls;
import gamlss.utilities.MakeLinkFunction;

import java.util.Hashtable;
import org.apache.commons.math3.distribution.UniformRealDistribution;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.special.Beta;
import org.apache.commons.math3.special.Gamma;
import org.apache.commons.math3.stat.descriptive.moment.Mean;
import org.apache.commons.math3.stat.descriptive.moment.StandardDeviation;
import org.apache.commons.math3.util.FastMath;

/**
 * @author Dr. Vlasios Voudouris, Daniil Kiose, 
 * Prof. Mikis Stasinopoulos and Dr Robert Rigby.
 */
public class SST implements GAMLSSFamilyDistribution {
	
	/** Number of distribution parameters. */
	private final int numDistPar = 4;
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
	/** vector of values of tau distribution parameter. */
	private ArrayRealVector tauV;
	/** Array of first derrivative values dl/dmu. */
	private double[] dldm;
	/** Array of first derrivative values dl/dsigma. */
	private double[] dlds;
	/** Array of first derrivative values dl/dnu. */
	private double[] dldn;
	/** Array of first derrivative values dl/dtau. */
	private double[] dldt;
	/** Temporary vector for interim operations. */
	private ArrayRealVector tempV;	
	/** Temporary array for interim operations. */
	private double[] m1;
	/** Temporary array for interim operations. */
	private double[] m2;
	/** Temporary array for interim operations. */
	private double[] s1;
	/** Temporary array for interim operations. */
	private double[] mu1;
	/** Temporary array for interim operations. */
	private double[] sigma1;
	/** Temporary int for interim operations. */
	private int size;
	/** Object of ST3 distribution class. */
	private ST3 st3 = new ST3();

	
	/** This is the Standardized skew t  distribution with default
	 *  link (muLink="identity",sigmaLink="log", nuLink="log",
	 *   tauLink="logShiftTo2"). */
	public SST() {
	
	this(DistributionSettings.IDENTITY, 
		 DistributionSettings.LOG, 
		 DistributionSettings.LOG, 
		 DistributionSettings.LOGSHIFTTO2);
	}
	
	/** This is the Standardized skew t distribution with
	 *  supplied link function for each of the distribution parameters. 
	 * @param muLink - link function for mu distribution parameter
	 * @param sigmaLink - link function for sigma distribution parameter
	 * @param nuLink - link function for nu distribution parameter
	 * @param tauLink - link function for tau distribution parameter*/
	public SST(final int muLink, 
			   final int sigmaLink, 
			   final int nuLink, 
			   final int tauLink) {
	
		distributionParameterLink.put(DistributionSettings.MU, 	   
			MakeLinkFunction.checkLink(DistributionSettings.SST , muLink));
		distributionParameterLink.put(DistributionSettings.SIGMA,  
			MakeLinkFunction.checkLink(DistributionSettings.SST, sigmaLink));
		distributionParameterLink.put(DistributionSettings.NU,     
			MakeLinkFunction.checkLink(DistributionSettings.SST, nuLink));
		distributionParameterLink.put(DistributionSettings.TAU,    
			MakeLinkFunction.checkLink(DistributionSettings.SST, tauLink));
		
		st3 = new ST3();
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
		distributionParameters.put(DistributionSettings.TAU,
													setTauInitial(y));
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
		//nu.initial = expression(nu <- rep(1, length(y))),
		tempV = new ArrayRealVector(y.getDimension());
		tempV.set(1.0);
		return tempV;
	}


	/** Calculates initial value of tau.
	 * @param y - vector of values of response variable
	 * @return vector of initial values of tau
	 */
	private ArrayRealVector setTauInitial(final ArrayRealVector y) {
		//tau.initial = expression(tau <-rep(4, length(y)))
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
	    case DistributionSettings.TAU:
	    	tempV = dldt(y);
	      break;
		default: 
			System.err.println("Requested first order "
					+ "derivative does not exist");
			break;
	    }			
		return tempV;
	}
	
	/**
	 * Set m1, m2, s1, mu1 arrays.
	 * @param y - response variable
	 */
	private void setInterimArrays(final ArrayRealVector y) {
		muV     = distributionParameters.get(DistributionSettings.MU);
		sigmaV  = distributionParameters.get(DistributionSettings.SIGMA);
	 	nuV     = distributionParameters.get(DistributionSettings.NU);
	 	tauV    = distributionParameters.get(DistributionSettings.TAU);
		
	 	size = y.getDimension();
		m1 = new double[size];
		m2 = new double[size];
		s1 = new double[size];
		mu1 = new double[size];
		sigma1 = new double[size];
	 	for (int i = 0; i < size; i++) {
	 		
			//m1 <- (2*tau^(1/2)*(nu^2-1))/((tau-1)*beta(1/2, tau/2)*nu)
			m1[i] = (2 * (FastMath.sqrt(tauV.getEntry(i))) 
					* (nuV.getEntry(i) * nuV.getEntry(i) - 1)) 
								/ ((tauV.getEntry(i) - 1) 
								* FastMath.exp(Beta.logBeta(0.5, 
									tauV.getEntry(i) / 2)) 
											* nuV.getEntry(i));
			
			//m2 <- (tau*(nu^3+(1/nu^3)))/((tau-2)*(nu+(1/nu)))
			m2[i] = (tauV.getEntry(i) * (nuV.getEntry(i) 
					* nuV.getEntry(i) * nuV.getEntry(i) 
					+ (1 / (nuV.getEntry(i) * nuV.getEntry(i) 
							* nuV.getEntry(i))))) / ((tauV.getEntry(i) 
										- 2) * (nuV.getEntry(i) + (1 
													/ nuV.getEntry(i))));
			
			//s1 <- sqrt(m2-m1^2)
			s1[i] = FastMath.sqrt(m2[i] - m1[i] * m1[i]);
			
			//mu1 <- mu- ((sigma*m1)/s1)
			mu1[i] = muV.getEntry(i) - ((sigmaV.getEntry(i) * m1[i]) / s1[i]);
			
			//sigma1 <- sigma/s1
			sigma1[i] = sigmaV.getEntry(i) / s1[i];
	 	}
	}
	

	/**  First derivative dldm = dl/dmu, where l - log-likelihood function.
	 * @param y - vector of values of response variable
	 * @return  a vector of first derivative dldm = dl/dmu
	 */	
	public final ArrayRealVector dldm(final ArrayRealVector y) { 

	 	//dldm <- ST3()$dldm(y, mu1, sigma1, nu, tau)
		st3.setDistributionParameter(DistributionSettings.MU, 
									new ArrayRealVector(mu1, false));
		st3.setDistributionParameter(DistributionSettings.SIGMA, 
									new ArrayRealVector(sigma1, false));
		st3.setDistributionParameter(DistributionSettings.NU, nuV);
		st3.setDistributionParameter(DistributionSettings.TAU, tauV);
		
		tempV = st3.firstDerivative(DistributionSettings.MU, y);
		dldm = tempV.getDataRef();
		
		m1 = null;
		m2 = null;
		s1 = null;
		mu1 = null;
		sigma1 = null;
		return tempV;
	}


	/** First derivative dlds = dl/dsigma, where l - log-likelihood function.
	 * @param y - vector of values of response variable
	 * @return  a vector of First derivative dlds = dl/dsigma
	 */	
	public final ArrayRealVector dlds(final ArrayRealVector y) {
	
		//dldd <- -(m1/s1)*ST3()$dldm(y, mu1, sigma1, nu, tau) 
		//+ (1/s1)*ST3()$dldd(y, mu1, sigma1, nu, tau)
		st3.setDistributionParameter(DistributionSettings.MU, 
						new ArrayRealVector(mu1, false));
		st3.setDistributionParameter(DistributionSettings.SIGMA, 
						new ArrayRealVector(sigma1, false));
		st3.setDistributionParameter(DistributionSettings.NU, nuV);
		st3.setDistributionParameter(DistributionSettings.TAU, tauV);
		
		double[] dldmTemp 
				= st3.firstDerivative(DistributionSettings.MU, y).getDataRef();
		double[] dldsTemp 
			= st3.firstDerivative(DistributionSettings.SIGMA, y).getDataRef();
		
		dlds = new double[size];
	 	for (int i = 0; i < size; i++) {
	 		dlds[i] = -(m1[i] / s1[i]) * dldmTemp[i] 
	 									+ (1 / s1[i]) * dldsTemp[i];
	 	}

		m1 = null;
		m2 = null;
		s1 = null;
		mu1 = null;
		sigma1 = null;
		return new ArrayRealVector(dlds, false);
		}
	

	/** First derivative dldn = dl/dnu, where l - log-likelihood function.
	 * @param y - vector of values of response variable
	 * @return  a vector of First derivative dldn = dl/dnu
	 */	
	public final ArrayRealVector dldn(final ArrayRealVector y) {
		
		st3.setDistributionParameter(DistributionSettings.MU, 
						new ArrayRealVector(mu1, false));
		st3.setDistributionParameter(DistributionSettings.SIGMA, 
						new ArrayRealVector(sigma1, false));
		st3.setDistributionParameter(DistributionSettings.NU, nuV);
		st3.setDistributionParameter(DistributionSettings.TAU, tauV);
		
        //dl1dmu1 <- ST3()$dldm(y, mu1, sigma1, nu, tau)
        //dl1dd1 <- ST3()$dldd(y, mu1, sigma1, nu, tau)
        //dl1dv <- ST3()$dldv(y, mu1, sigma1, nu, tau)
		double[] dl1dmu1 
				= st3.firstDerivative(DistributionSettings.MU, y).getDataRef();
		double[] dl1dd1 
			 = st3.firstDerivative(DistributionSettings.SIGMA, y).getDataRef();
		double[] dl1dv 
			 = st3.firstDerivative(DistributionSettings.NU, y).getDataRef();
		
		dldn = new double[size];
	 	for (int i = 0; i < size; i++) {
	 		
	 		//dmu1dm1 <- -sigma/s1
	 		final double dmu1dm1 = -(sigmaV.getEntry(i) / s1[i]);
	 		
            //dmu1ds1 <- (sigma*m1)/(s1^2)
	 		final double dmu1ds1 = (sigmaV.getEntry(i) 
	 										* m1[i]) / (s1[i] * s1[i]);
	 		
            //dd1ds1 <- -sigma/(s1^2)
	 		final double dd1ds1 = -(sigmaV.getEntry(i) / (s1[i] * s1[i])); 
	 		
            //dm1dv <- ((2*tau^(1/2))/((tau-1)
	 		//*beta(1/2, tau/2)))*((nu^2+1)/(nu^2))
	 		final double dm1dv = ((2 * FastMath.sqrt(tauV.getEntry(i))) 
	 				/ ((tauV.getEntry(i) - 1) * FastMath.exp(
	 						Beta.logBeta(0.5, tauV.getEntry(i) 
	 								/ 2)))) * ((nuV.getEntry(i) 
	 										* nuV.getEntry(i) + 1)
	 										/ (nuV.getEntry(i) 
	 												* nuV.getEntry(i)));
	 		
            //dm2dv <- m2*((6*nu^5/(nu^6+1))-(2/nu)-(2*nu/(nu^2+1)))
	 		final double dm2dv = m2[i] * ((6.0 
	 				* FastMath.pow(nuV.getEntry(i), 5) / (
	 						FastMath.pow(nuV.getEntry(i), 6) 
	 						+ 1)) - (2.0 / nuV.getEntry(i)) 
	 						- (2.0 * nuV.getEntry(i) / (nuV.getEntry(i) 
	 										* nuV.getEntry(i) + 1.0)));
	 		
            //ds1dv <- (dm2dv - 2*m1*dm1dv)/(2*s1)
	 		final double ds1dv = (dm2dv - 2.0 * m1[i] * dm1dv) / (2 * s1[i]);
	 		
            //dldv <- dl1dmu1*dmu1dm1*dm1dv 
	 		//+ dl1dmu1*dmu1ds1*ds1dv + dl1dd1*dd1ds1*ds1dv + dl1dv
	 		dldn[i] = dl1dmu1[i] * dmu1dm1 * dm1dv + dl1dmu1[i] * dmu1ds1 
	 						* ds1dv + dl1dd1[i] * dd1ds1 * ds1dv + dl1dv[i]; 
		 }

	 	m1 = null;
	 	m2 = null;
	 	s1 = null;
	 	mu1 = null;
	 	sigma1 = null;
		return new ArrayRealVector(dldn, false);
	}
	
	/** First derivative dldtau = dl/dtau, where l - log-likelihood function.
	 * @param y - vector of values of response variable
	 * @return  a vector of First derivative dldtau = dl/dtau
	 */
	public final ArrayRealVector dldt(final ArrayRealVector y) {
		
		st3.setDistributionParameter(DistributionSettings.MU, 
						new ArrayRealVector(mu1, false));
		st3.setDistributionParameter(DistributionSettings.SIGMA, 
						new ArrayRealVector(sigma1, false));
		st3.setDistributionParameter(DistributionSettings.NU, nuV);
		st3.setDistributionParameter(DistributionSettings.TAU, tauV);
		
		//dl1dmu1 <- ST3()$dldm(y, mu1, sigma1, nu, tau)
		//dl1dd1 <- ST3()$dldd(y, mu1, sigma1, nu, tau)
		//dl1dt <- ST3()$dldt(y, mu1, sigma1, nu, tau)
		double[] dl1dmu1 
				= st3.firstDerivative(DistributionSettings.MU, y).getDataRef();
		double[] dl1dd1 
			 = st3.firstDerivative(DistributionSettings.SIGMA, y).getDataRef();
		double[] dl1dt 
			 = st3.firstDerivative(DistributionSettings.TAU, y).getDataRef();
		
		dldt = new double[size];
	 	for (int i = 0; i < size; i++) {
	 		
            //dmu1dm1 <- -sigma/s1
	 		final double dmu1dm1 = -(sigmaV.getEntry(i) / s1[i]);
	 		
            //dmu1ds1 <- (sigma*m1)/(s1^2)
	 		final double dmu1ds1 = (sigmaV.getEntry(i)  
	 									* m1[i]) / (s1[i] * s1[i]);
	 		
            //dd1ds1 <- -sigma/(s1^2)
	 		final double dd1ds1 = -(sigmaV.getEntry(i) / (s1[i] * s1[i]));
	 		
            //dm1dt <- m1*((1/(2*tau))-(1/(tau-1))- 0.5*(digamma(tau/2)) 
	 		//+ 0.5*(digamma((tau+1)/2)))
	 		final double dm1dt = m1[i] * ((1 / (2 * tauV.getEntry(i)))
	 				- (1 / (tauV.getEntry(i) - 1)) - 0.5 
	 					* (Gamma.digamma(tauV.getEntry(i) / 2))  
	 					+ 0.5 * (Gamma.digamma((tauV.getEntry(i) + 1) / 2)));
	 		
            //dm2dt <- -m2*(2/(tau*(tau-2))) 
	 		final double dm2dt = -m2[i] * (2 / (tauV.getEntry(i) 
	 												* (tauV.getEntry(i) - 2)));
	 		
            //ds1dt <- (dm2dt - 2*m1*dm1dt)/(2*s1)
	 		final double ds1dt = (dm2dt - 2.0 * m1[i] * dm1dt) / (2 * s1[i]);
	 		
            //dldt <- dl1dmu1*dmu1dm1*dm1dt + dl1dmu1*dmu1ds1
	 		//*ds1dt + dl1dd1*dd1ds1*ds1dt + dl1dt
	 		dldt[i] = dl1dmu1[i] * dmu1dm1 * dm1dt + dl1dmu1[i] * dmu1ds1 
	 						   * ds1dt + dl1dd1[i] * dd1ds1 * ds1dt + dl1dt[i];
	 	}

	 	m1 = null;
	 	m2 = null;
	 	s1 = null;
	 	mu1 = null;
	 	sigma1 = null;
		return new ArrayRealVector(dldt, false);
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
			      case DistributionSettings.TAU:
			    	 tempV = d2ldt2(y);
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
	 		//d2ldm2 <- -dldm*dldm
	 		out[i] = -dldm[i] * dldm[i];
	 	}
	 	muV     = null;
	 	sigmaV  = null;
	 	nuV     = null;
	 	tauV    = null;
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
	 		//d2ldd2 <- -dldd*dldd
	 		out[i] = -dlds[i] * dlds[i];
	 	 }
	 	 muV     = null;
	 	 sigmaV  = null;
	 	 nuV     = null;
	 	 tauV    = null;
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
	 		//d2ldv2 <- -dldv*dldv
	 		out[i] = -dldn[i] * dldn[i];
	 	 }
	 	 muV     = null;
	 	 sigmaV  = null;
	 	 nuV     = null;
	 	 tauV    = null;
	  	return new ArrayRealVector(out, false);
	}
	
	/** Second derivative d2ldt2= (d^2l)/(dtau^2),
	 * where l - log-likelihood function.
	 * @param y - vector of values of response variable
	 * @return  a vector of second derivative d2ldt2= (d^2l)/(dtau^2)
	 */ 
	private ArrayRealVector d2ldt2(final ArrayRealVector y) {	
		
		double[] out = new double[size];
	 	for (int i = 0; i < size; i++) {
	 		//d2ldt2 <- -dldt*dldt
	 		out[i] = -dldt[i] * dldt[i];
	 	 }
	 	 muV     = null;
	 	 sigmaV  = null;
	 	 nuV     = null;
	 	 tauV    = null;
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
			case DistributionSettings.TAU:
				tempV = d2ldmdt(y);
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
			case DistributionSettings.TAU:
				tempV = d2ldsdt(y);
				break;
	          default: 
		  		System.err.println("Second derivative does not exist");
		  		return null;
			}
		}
		if (whichDistParameter1 == DistributionSettings.NU) {
			switch (whichDistParameter2) {
			case DistributionSettings.TAU:
				tempV = d2ldndt(y);
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
		double[] out = new double[size];
	 	for (int i = 0; i < size; i++) {
	 		// d2ldmdd <- -(dldm*dldd)
	 		out[i] = -dldm[i] * dlds[i];
	 	}
	  	return new ArrayRealVector(out, false);
	}

	/** Second cross derivative of likelihood function
 	 * in respect to mu and nu (d2ldmdd = d2l/dmu*dnu).
	 * @param y - vector of values of response variable
	 * @return  a vector of Second cross derivative
	 */
	private ArrayRealVector d2ldmdn(final ArrayRealVector y) {
		double[] out = new double[size];
	 	for (int i = 0; i < size; i++) {
	 		//d2ldmdv <- -(dldm*dldv)
	 		out[i] = -dldm[i] * dldn[i];
	 	}
	  	return new ArrayRealVector(out, false);
	}	
	
	/** Second cross derivative of likelihood function 
	 * in respect to mu and tau (d2ldmdd = d2l/dmu*dtau).
	 * @param y - vector of values of response variable
	 * @return  a vector of Second cross derivative
	 */
	private ArrayRealVector d2ldmdt(final ArrayRealVector y) {	
		double[] out = new double[size];
	 	for (int i = 0; i < size; i++) {
	 		//d2ldmdt <- -(dldm*dldt)
	 		out[i] = -dldm[i] * dldt[i];
	 	}
	  	return new ArrayRealVector(out, false);
	}  	
	
	/** Second cross derivative of likelihood function 
	 * in respect to sigma and nu (d2ldmdd = d2l/dsigma*dnu).
	 * @param y - vector of values of response variable
	 * @return  a vector of Second cross derivative
	 */
	private ArrayRealVector d2ldsdn(final ArrayRealVector y) {	
		double[] out = new double[size];
	 	for (int i = 0; i < size; i++) {
	 		//d2ldddv <- -(dldd*dldv)
	 		out[i] = -dlds[i] * dldn[i];
	 	}
	  	return new ArrayRealVector(out, false);
	}
	
	/** Second cross derivative of likelihood function 
	 * in respect to sigma and tau (d2ldmdd = d2l/dsigma*dtau).
	 * @param y - vector of values of response variable
	 * @return  a vector of Second cross derivative
	 */
	private ArrayRealVector d2ldsdt(final ArrayRealVector y) {	
		double[] out = new double[size];
	 	for (int i = 0; i < size; i++) {
	 		//d2ldddt <- -(dldd*dldt) 
	 		out[i] = -dlds[i] * dldt[i];
	 	}
	  	return new ArrayRealVector(out, false);
	}	
	

	/** Second cross derivative of likelihood function 
	 * in respect to nu and tau (d2ldmdd = d2l/dnu*dtau).
	 * @param y - vector of values of response variable
	 * @return  a vector of Second cross derivative
	 */
	private ArrayRealVector d2ldndt(final ArrayRealVector y) {	
		double[] out = new double[size];
	 	for (int i = 0; i < size; i++) {
	 		//d2ldvdt <- -(dldv*dldt)
	 		out[i] = -dldn[i] * dldt[i];
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
		double[] tauArr = distributionParameters.get(
									DistributionSettings.TAU).getDataRef();
		
		for (int i = 0; i < size; i++) {
			
			out[i] = (-2) * dSST(y.getEntry(i), muArr[i], sigmaArr[i], 
								nuArr[i], tauArr[i], Controls.LOG_LIKELIHOOD);
		}
		return new ArrayRealVector(out, false);
	}
	
		/** Computes the probability density function (PDF) of this 
		 * distribution evaluated at the specified point x.
		 * @param x - value of response variable
		 * @param mu - value of mu distribution parameter
		 * @param sigma - value of sigma distribution parameter
		 * @param nu - value of nu distribution parameter
		 * @param tau - value of tau distribution parameter
		 * @param isLog  - logical, whether to take log of the function or not
		 * @return value of probability density function
		 */
		public final double dSST(final double x, 
								 final double mu, 
								 final double sigma, 
								 final double nu, 
								 final double tau, 
								 final boolean isLog) {
		
 		//m1 <- (2*tau^(1/2)*(nu^2-1))/((tau-1)*beta(1/2, tau/2)*nu)
		final double m1 = (2 * (FastMath.sqrt(tau)) * (nu 
					* nu - 1)) / ((tau - 1) * FastMath.exp(
								Beta.logBeta(0.5, tau / 2)) * nu);
		
		//m2 <- (tau*(nu^3+(1/nu^3)))/((tau-2)*(nu+(1/nu)))
		final double m2 = (tau * (nu * nu * nu + (1 / (nu * nu 
						* nu)))) / ((tau - 2) * (nu + (1 / nu)));
		
		//s1 <- sqrt(m2-m1^2)
		final double s1 = FastMath.sqrt(m2 - m1 * m1);
		
		//mu1 <- mu- ((sigma*m1)/s1)
		final double mu1 = mu - ((sigma * m1) / s1);
		
		//sigma1 <- sigma/s1
		final double sigma1 = sigma / s1;
	 	
		return st3.dST3(x, mu1, sigma1, nu, tau, isLog);
	}
	
	/**
	 * dSST(x) launches dSST(x, mu, sigma, nu, isLog) 
	 * with deafult mu=0, sigma=1, nu=0.8, tau=7, isLog = false.
	 * @param x - value of response variable 
	 * @return value of probability density function 
	 */
	//pSST <- (x, mu=0, sigma=1, nu=0.8, tau=7, log = FALSE)
	public final double dSST(final double x) {
		return dSST(x, 0.0, 1.0, 0.8, 7.0, false);
	}

	/** Computes the cumulative distribution 
	 * function P(X <= q) for a random variable X .
	 * whose values are distributed according to this distribution
	 * @param q - value of quantile
	 * @param mu - value of mu distribution parameter
	 * @param sigma - value of sigma distribution parameter
	 * @param nu - value of nu distribution parameter 
	 * @param tau - value of tau distribution parameter 
	 * @param lowerTail - logical, if TRUE (default), probabilities
	 *  are P[X <= x] otherwise, P[X > x].
	 * @param isLog - logical, if TRUE, probabilities p are given as log(p)
	 * @return value of cumulative probability function values P(X <= q)
	 */
	public final double pSST(final double q, 
 	   		 				 final double mu, 
 	   		 				 final double sigma, 
 	   		 				 final double nu,
 	   		 				 final double tau, 
 	   		 				 final boolean lowerTail, 
 	   		 				 final boolean isLog) {

 		//m1 <- (2*tau^(1/2)*(nu^2-1))/((tau-1)*beta(1/2, tau/2)*nu)
		final double m1 = (2 * (FastMath.sqrt(tau)) * (nu 
					* nu - 1)) / ((tau - 1) * FastMath.exp(
								Beta.logBeta(0.5, tau / 2)) * nu);
		
		//m2 <- (tau*(nu^3+(1/nu^3)))/((tau-2)*(nu+(1/nu)))
		final double m2 = (tau * (nu * nu * nu + (1 / (nu * nu 
						* nu)))) / ((tau - 2) * (nu + (1 / nu)));
		
		//s1 <- sqrt(m2-m1^2)
		final double s1 = FastMath.sqrt(m2 - m1 * m1);
		
		//mu1 <- mu- ((sigma*m1)/s1)
		final double mu1 = mu - ((sigma * m1) / s1);
		
		//sigma1 <- sigma/s1
		final double sigma1 = sigma / s1;
	 	
		return st3.pST3(q, mu1, sigma1, nu, tau, lowerTail, isLog);
	}
	
	/**
	 * pSST(q) launches pSST(q, mu, sigma, nu,  tau, lowerTail, isLog) 
	 * with deafult mu=0, sigma=1, nu=0.8, tau=7.
	 * lowerTail = true, isLog = false.
	 * @param q - vector of quantiles
	 * @return vector of cumulative probability function values P(X <= q)
	 */
	//pSST <- (q, mu=0, sigma=1, nu=0.8, tau=7, 
	//lower.tail = TRUE, log.p = FALSE){
	public final double pSST(final double q) {
		return pSST(q,  0.0, 1.0, 0.8, 7.0, true, false);
	}	
	
	/** Computes the quantile (inverse cumulative probability)
	 *  function  of this distribution.
	* @param p - value of cumulative probability
	* @param mu -  value of mu distribution parameters
	* @param sigma -  value of sigma distribution parameters
	* @param nu -  value of nu distribution parameters 
	* @param tau -  value of tau distribution parameters
	* @param lowerTail - logical; if TRUE (default), probabilities 
	* are P[X <= x] otherwise, P[X > x]
	* @param isLog - logical; if TRUE, probabilities p are given as log(p).
	* @return value of quantile function
	*/
	public final double qSST(final double p, 
	         				 final double  mu, 
	         				 final double  sigma, 
	         				 final double nu, 
	         				 final double tau, 
	         				 final boolean lowerTail, 
	         				 final boolean isLog) {
		
		//if (any(tau <= 2)) stop(paste("tau must be greater than 2"))
		if (tau <= 2) {
			System.err.println("tau must be greater than 2");
			return -1.0;
		}
		
 		//m1 <- (2*tau^(1/2)*(nu^2-1))/((tau-1)*beta(1/2, tau/2)*nu)
		final double m1 = (2 * (FastMath.sqrt(tau)) * (nu 
					* nu - 1)) / ((tau - 1) * FastMath.exp(
								Beta.logBeta(0.5, tau / 2)) * nu);
		
		//m2 <- (tau*(nu^3+(1/nu^3)))/((tau-2)*(nu+(1/nu)))
		final double m2 = (tau * (nu * nu * nu + (1 / (nu * nu 
						* nu)))) / ((tau - 2) * (nu + (1 / nu)));
		
		//s1 <- sqrt(m2-m1^2)
		final double s1 = FastMath.sqrt(m2 - m1 * m1);
		
		//mu1 <- mu- ((sigma*m1)/s1)
		final double mu1 = mu - ((sigma * m1) / s1);
		
		//sigma1 <- sigma/s1
		final double sigma1 = sigma / s1;
	 	
		return st3.qST3(p, mu1, sigma1, nu, tau, lowerTail, isLog);
	}
	

	/**
	 * qSST(p) launches qSST(p, mu, sigma, nu,  tau, lowerTail, isLog)
	 *  with deafult mu=0, sigma=1, nu=0.8, tau=7.
	 * lower.tail = TRUE, log.p = FALSE 
	 * @param p - value of cumulative probability 
	 * @return quantile
	 */
	//qSST <- function(p, mu=0, sigma=1, nu=0.8, tau=7, lower.tail = TRUE,
	public final double qSST(final double p) {
		return qSST(p, 0.0, 1.0, 0.8, 7.0, true, false);
	}

	/** Generates a random sample from this distribution.
	 * @param mu -  vector of mu distribution parameters values
	 * @param sigma -  vector of sigma distribution parameters values
	 * @param nu -  vector of nu distribution parameters values
	 * @param tau -  vector of tau distribution parameters values
	 * @param uDist -  object of UniformRealDistribution class;
	 * @return random sample vector
	 */	
		public final double rSST(final double mu, 
		       	   				 final double sigma, 
		       	   				 final double nu, 
		       	   				 final double tau,
		       	   				 final UniformRealDistribution uDist) {	
	
			//if (any(tau <= 2)) stop(paste("tau must be greater than 2")
			if (tau <= 2) {
				System.err.println("tau must be greater than 2");
				return -1.0;
			}
			//r <- qST3(p,mu=mu,sigma=sigma,nu=nu,tau=tau)
			return qSST(uDist.sample(), mu, sigma, nu, tau, true, false);
		}
	
	/**
	* rSST() launches rSST(mu, sigma, nu,  tau, uDist) 
	* with deafult mu=0, sigma=1, nu=0.8, tau=7.
	 * @param uDist -  object of UniformRealDistribution class;
	 * @return random sample vector
	*/
	//rSST <- function(n, mu=0, sigma=1, nu=0.8, tau=7){
	public final double rSST(final UniformRealDistribution uDist) {
		return rSST(0.0, 1.0, 0.8, 7.0, uDist);
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
      case DistributionSettings.TAU:
      	tempB = isTauValid(
      				distributionParameters.get(DistributionSettings.TAU));
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
	 * Checks whether the tau distribution parameter is valid.
	 * @param tau - vector of nu values
	 * @return - - boolean
	 */
	private boolean isTauValid(final ArrayRealVector tau) {
		return tau.getMinValue() > 0;	
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
		return DistributionSettings.SST;
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
