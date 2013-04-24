package gamlss.utilities;
	/*
	  Copyright 2012 by Dr. Vlasios Voudouris and ABM Analytics Ltd
	  Licensed under the Academic Free License version 3.0
	  See the file "LICENSE" for more information
	*/

	import gamlss.distributions.*;
	import gamlss.utilities.Controls;
	import gamlss.utilities.MatrixFunctions;
	import gamlss.utilities.oi.CSVFileReader;

	import java.util.ArrayList;
	import java.util.HashMap;
	import java.util.Hashtable;

	import org.apache.commons.math3.linear.ArrayRealVector;
	import org.apache.commons.math3.linear.BlockRealMatrix;

	public class TestingDistributions {
		
		public TestingDistributions(){
			//twoParTest();
			//threeParTest();
			fourParTest();
		}
		

		 
		private void twoParTest() {
				
				 String fileName = "Data/distTest.csv";
				 CSVFileReader readData = new CSVFileReader(fileName);
				 readData.readFile();
				 ArrayList<String> data = readData.storeValues;
				 
				 ArrayRealVector y = new ArrayRealVector(data.size());
				 ArrayRealVector mu = new ArrayRealVector(data.size());
				 ArrayRealVector sigma = new ArrayRealVector(data.size());
				 ArrayRealVector nu = new ArrayRealVector(data.size());
				 ArrayRealVector tau = new ArrayRealVector(data.size());
				 

				
				 for (int i = 0; i < data.size(); i++) {
					String[] line = data.get(i).split(",");
					y.setEntry(i,  Double.parseDouble(line[0]));
					mu.setEntry(i,  Double.parseDouble(line[1]));
					sigma.setEntry(i,  Double.parseDouble(line[2]));
					nu.setEntry(i,  Double.parseDouble(line[3]));
					tau.setEntry(i,  Double.parseDouble(line[4]));
				 }	
				 
				 double[] outA = new double[y.getDimension()];
				String folder = "C:\\Users\\Daniil\\Desktop\\Gamlss_exp/outDistG.csv";
				ArrayRealVector out = null;
				GA dist = new GA();
				
				dist.setDistributionParameter(DistributionSettings.MU, mu);
				dist.setDistributionParameter(DistributionSettings.SIGMA, sigma);
				dist.setDistributionParameter(DistributionSettings.NU, nu);
				dist.setDistributionParameter(DistributionSettings.TAU, tau);
				//1		
				out = dist.firstDerivative(DistributionSettings.MU, y);
				MatrixFunctions.vectorWriteCSV(folder, out, false);
				//2
				out = dist.secondDerivative(DistributionSettings.MU, y);
				MatrixFunctions.vectorWriteCSV(folder, out, true);
				//3
				out = dist.firstDerivative(DistributionSettings.SIGMA, y);
				MatrixFunctions.vectorWriteCSV(folder, out, true);
				//4
				out = dist.secondDerivative(DistributionSettings.SIGMA, y);
				MatrixFunctions.vectorWriteCSV(folder, out, true);
				//5
//				out = dist.secondCrossDerivative(DistributionSettings.MU, DistributionSettings.SIGMA,  y);
//				MatrixFunctions.vectorWriteCSV(folder, out, true);
				//6
				for(int i = 0; i < y.getDimension(); i++) { 
					outA[i] = dist.dGA(y.getEntry(i), mu.getEntry(i), sigma.getEntry(i), false);
				}
				MatrixFunctions.vectorWriteCSV(folder, new ArrayRealVector(outA, false), true);
				//7
				for(int i = 0; i < y.getDimension(); i++) { 
					outA[i] = dist.dGA(y.getEntry(i), mu.getEntry(i), sigma.getEntry(i), true);
				}
				MatrixFunctions.vectorWriteCSV(folder, new ArrayRealVector(outA, false), true);
				//8
				for(int i = 0; i < y.getDimension(); i++) { 
					outA[i] = dist.pGA(y.getEntry(i), mu.getEntry(i), sigma.getEntry(i), true, true);
				}
				MatrixFunctions.vectorWriteCSV(folder, new ArrayRealVector(outA, false), true);
				//9
				for(int i = 0; i < y.getDimension(); i++) { 
					outA[i] = dist.pGA(y.getEntry(i), mu.getEntry(i), sigma.getEntry(i), false, true);
				}
				MatrixFunctions.vectorWriteCSV(folder, new ArrayRealVector(outA, false), true);
				//10
				for(int i = 0; i < y.getDimension(); i++) { 
					outA[i] = dist.pGA(y.getEntry(i), mu.getEntry(i), sigma.getEntry(i), true, false);
				}
				MatrixFunctions.vectorWriteCSV(folder, new ArrayRealVector(outA, false), true);
				//11
				
				for(int i = 0; i < y.getDimension(); i++) { 
					outA[i] = dist.pGA(y.getEntry(i), mu.getEntry(i), sigma.getEntry(i), false, false);
				}
				MatrixFunctions.vectorWriteCSV(folder, new ArrayRealVector(outA, false), true);
				//12
				for(int i = 0; i < y.getDimension(); i++) { 
//					outA[i] = dist.qGA(y.getEntry(i), mu.getEntry(i), sigma.getEntry(i), true, true);
				}
//				MatrixFunctions.vectorWriteCSV(folder, new ArrayRealVector(outA, false), true);
				//13
				for(int i = 0; i < y.getDimension(); i++) { 
//					outA[i] = dist.qGA(y.getEntry(i), mu.getEntry(i), sigma.getEntry(i), false, true);
				}
//				MatrixFunctions.vectorWriteCSV(folder, new ArrayRealVector(outA, false), true);
				//14
				for(int i = 0; i < y.getDimension(); i++) { 
					outA[i] = dist.qGA(y.getEntry(i), mu.getEntry(i), sigma.getEntry(i), true, false);
				}
				MatrixFunctions.vectorWriteCSV(folder, new ArrayRealVector(outA, false), true);
				//15
				for(int i = 0; i < y.getDimension(); i++) { 
					outA[i] = dist.qGA(y.getEntry(i), mu.getEntry(i), sigma.getEntry(i), false, false);
				}
				MatrixFunctions.vectorWriteCSV(folder, new ArrayRealVector(outA, false), true);
				//16
				for(int i = 0; i < y.getDimension(); i++) { 
					outA[i] = dist.dGA(y.getEntry(i));
				}
//				MatrixFunctions.vectorWriteCSV(folder, new ArrayRealVector(outA, false), true);
				//17
				for(int i = 0; i < y.getDimension(); i++) { 
					outA[i] = dist.pGA(y.getEntry(i));
				}
//				MatrixFunctions.vectorWriteCSV(folder, new ArrayRealVector(outA, false), true);
				//18
				for(int i = 0; i < y.getDimension(); i++) { 
					outA[i] = dist.qGA(y.getEntry(i));
				}
//				MatrixFunctions.vectorWriteCSV(folder, new ArrayRealVector(outA, false), true);
				
			
				System.out.println("Done !!!");
				
			}
		
		private void fourParTest() {
			
			 String fileName = "Data/distTest2.csv";
			 CSVFileReader readData = new CSVFileReader(fileName);
			 readData.readFile();
			 ArrayList<String> data = readData.storeValues;
			 
			 ArrayRealVector y = new ArrayRealVector(data.size());
			 ArrayRealVector mu = new ArrayRealVector(data.size());
			 ArrayRealVector sigma = new ArrayRealVector(data.size());
			 ArrayRealVector nu = new ArrayRealVector(data.size());
			 ArrayRealVector tau = new ArrayRealVector(data.size());
			 

			
			 for (int i = 0; i < data.size(); i++) {
				String[] line = data.get(i).split(",");
				y.setEntry(i,  Double.parseDouble(line[0]));
				mu.setEntry(i,  Double.parseDouble(line[1]));
				sigma.setEntry(i,  Double.parseDouble(line[2]));
				nu.setEntry(i,  Double.parseDouble(line[3]));
				tau.setEntry(i,  Double.parseDouble(line[4]));
			 }	
		 
			double[] outA = new double[y.getDimension()];
			String folder = "C:\\Users\\Daniil\\Desktop\\Gamlss_exp/outDistG.csv";
			ArrayRealVector out = null;
			
			SST dist = new SST();
			
			dist.setDistributionParameter(DistributionSettings.MU, mu);
			dist.setDistributionParameter(DistributionSettings.SIGMA, sigma);
			dist.setDistributionParameter(DistributionSettings.NU, nu);
			dist.setDistributionParameter(DistributionSettings.TAU, tau);
			//0	
			out = dist.firstDerivative(DistributionSettings.MU, y);
			MatrixFunctions.vectorWriteCSV(folder, out, false);
			//1
			out = dist.secondDerivative(DistributionSettings.MU, y);
			MatrixFunctions.vectorWriteCSV(folder, out, true);
			//2
			out = dist.firstDerivative(DistributionSettings.SIGMA, y);
			MatrixFunctions.vectorWriteCSV(folder, out, true);
			//3
			out = dist.secondDerivative(DistributionSettings.SIGMA, y);
			MatrixFunctions.vectorWriteCSV(folder, out, true);
			//4
			out = dist.secondCrossDerivative(DistributionSettings.MU, DistributionSettings.SIGMA,  y);
			MatrixFunctions.vectorWriteCSV(folder, out, true);
			//5
			for(int i = 0; i < y.getDimension(); i++) { 
				outA[i] = dist.dSST(y.getEntry(i), mu.getEntry(i), sigma.getEntry(i), nu.getEntry(i), tau.getEntry(i),false);
			}
			MatrixFunctions.vectorWriteCSV(folder, new ArrayRealVector(outA, false), true);
			//6
			for(int i = 0; i < y.getDimension(); i++) { 
				outA[i] = dist.dSST(y.getEntry(i), mu.getEntry(i), sigma.getEntry(i),nu.getEntry(i),tau.getEntry(i), true);
			}
			MatrixFunctions.vectorWriteCSV(folder, new ArrayRealVector(outA, false), true);
			//7
			for(int i = 0; i < y.getDimension(); i++) { 
				outA[i] = dist.pSST(y.getEntry(i), mu.getEntry(i), sigma.getEntry(i),nu.getEntry(i), tau.getEntry(i),true, false);
				//System.out.println(outA[i]);
			}
			MatrixFunctions.vectorWriteCSV(folder, new ArrayRealVector(outA, false), true);
			//8
			for(int i = 0; i < y.getDimension(); i++) { 
				outA[i] = dist.pSST(y.getEntry(i), mu.getEntry(i), sigma.getEntry(i),nu.getEntry(i), tau.getEntry(i),false, true);
			}
			MatrixFunctions.vectorWriteCSV(folder, new ArrayRealVector(outA, false), true);
			//9
			for(int i = 0; i < y.getDimension(); i++) { 
				outA[i] = dist.pSST(y.getEntry(i), mu.getEntry(i), sigma.getEntry(i),nu.getEntry(i), tau.getEntry(i),true, false);
			}
			MatrixFunctions.vectorWriteCSV(folder, new ArrayRealVector(outA, false), true);
			//10
			for(int i = 0; i < y.getDimension(); i++) { 
				outA[i] = dist.pSST(y.getEntry(i), mu.getEntry(i), sigma.getEntry(i),nu.getEntry(i), tau.getEntry(i),false, false);
			}
			MatrixFunctions.vectorWriteCSV(folder, new ArrayRealVector(outA, false), true);
			//11
			for(int i = 0; i < y.getDimension(); i++) { 
//				outA[i] = dist.qSST(y.getEntry(i), mu.getEntry(i), sigma.getEntry(i),nu.getEntry(i),tau.getEntry(i), true, true);
			}
//			MatrixFunctions.vectorWriteCSV(folder, new ArrayRealVector(outA, false), true);
			//12
			for(int i = 0; i < y.getDimension(); i++) { 
//				outA[i] = dist.qSST(y.getEntry(i), mu.getEntry(i), sigma.getEntry(i),nu.getEntry(i),tau.getEntry(i), false, true);
			}
//			MatrixFunctions.vectorWriteCSV(folder, new ArrayRealVector(outA, false), true);
			//13
			for(int i = 0; i < y.getDimension(); i++) { 
				outA[i] = dist.qSST(y.getEntry(i), mu.getEntry(i), sigma.getEntry(i),nu.getEntry(i), tau.getEntry(i),true, false);
			}
			MatrixFunctions.vectorWriteCSV(folder, new ArrayRealVector(outA, false), true);
			//14
			for(int i = 0; i < y.getDimension(); i++) { 
				outA[i] = dist.qSST(y.getEntry(i), mu.getEntry(i), sigma.getEntry(i),nu.getEntry(i),tau.getEntry(i), false, false);
			}
			MatrixFunctions.vectorWriteCSV(folder, new ArrayRealVector(outA, false), true);
			//15
			out = dist.firstDerivative(DistributionSettings.NU, y);
			MatrixFunctions.vectorWriteCSV(folder, out, true);
			//16
			out = dist.secondDerivative(DistributionSettings.NU, y);
			MatrixFunctions.vectorWriteCSV(folder, out, true);
			//17
			out = dist.secondCrossDerivative(DistributionSettings.MU, DistributionSettings.NU,  y);
			MatrixFunctions.vectorWriteCSV(folder, out, true);
			//18
			out = dist.secondCrossDerivative(DistributionSettings.SIGMA, DistributionSettings.NU,  y);
			MatrixFunctions.vectorWriteCSV(folder, out, true);
			//19
			out = dist.firstDerivative(DistributionSettings.TAU, y);
			MatrixFunctions.vectorWriteCSV(folder, out, true);
			//20
			out = dist.secondDerivative(DistributionSettings.TAU, y);
			MatrixFunctions.vectorWriteCSV(folder, out, true);
			//21
			out = dist.secondCrossDerivative(DistributionSettings.MU, DistributionSettings.TAU,  y);
			MatrixFunctions.vectorWriteCSV(folder, out, true);
			//22
			out = dist.secondCrossDerivative(DistributionSettings.SIGMA, DistributionSettings.TAU,  y);
			MatrixFunctions.vectorWriteCSV(folder, out, true);
			//23
			out = dist.secondCrossDerivative(DistributionSettings.NU, DistributionSettings.TAU,  y);
			MatrixFunctions.vectorWriteCSV(folder, out, true);
		
			System.out.println("Done !!!");
			
		}
		
		private void threeParTest() {
			
			 String fileName = "Data/distTest.csv";
			 CSVFileReader readData = new CSVFileReader(fileName);
			 readData.readFile();
			 ArrayList<String> data = readData.storeValues;
			 
			 ArrayRealVector y = new ArrayRealVector(data.size());
			 ArrayRealVector mu = new ArrayRealVector(data.size());
			 ArrayRealVector sigma = new ArrayRealVector(data.size());
			 ArrayRealVector nu = new ArrayRealVector(data.size());
			 ArrayRealVector tau = new ArrayRealVector(data.size());
			 

			
			 for (int i = 0; i < data.size(); i++) {
				String[] line = data.get(i).split(",");
				y.setEntry(i,  Double.parseDouble(line[0]));
				mu.setEntry(i,  Double.parseDouble(line[1]));
				sigma.setEntry(i,  Double.parseDouble(line[2]));
				nu.setEntry(i,  Double.parseDouble(line[3]));
				tau.setEntry(i,  Double.parseDouble(line[4]));
			 }	
		 
			double[] outA = new double[y.getDimension()];
			String folder = "C:\\Users\\Daniil\\Desktop\\Gamlss_exp/outDistG.csv";
			ArrayRealVector out = null;
			
			TF2 dist = new TF2();
			
			dist.setDistributionParameter(DistributionSettings.MU, mu);
			dist.setDistributionParameter(DistributionSettings.SIGMA, sigma);
			dist.setDistributionParameter(DistributionSettings.NU, nu);
			dist.setDistributionParameter(DistributionSettings.TAU, tau);
			//1		
			out = dist.firstDerivative(DistributionSettings.MU, y);
			MatrixFunctions.vectorWriteCSV(folder, out, false);
			//2
			out = dist.secondDerivative(DistributionSettings.MU, y);
			MatrixFunctions.vectorWriteCSV(folder, out, true);
			//3
			out = dist.firstDerivative(DistributionSettings.SIGMA, y);
			MatrixFunctions.vectorWriteCSV(folder, out, true);
			//4
			out = dist.secondDerivative(DistributionSettings.SIGMA, y);
			MatrixFunctions.vectorWriteCSV(folder, out, true);
			//5
			out = dist.secondCrossDerivative(DistributionSettings.MU, DistributionSettings.SIGMA,  y);
			MatrixFunctions.vectorWriteCSV(folder, out, true);
			//6
			for(int i = 0; i < y.getDimension(); i++) { 
				outA[i] = dist.dTF2(y.getEntry(i), mu.getEntry(i), sigma.getEntry(i), nu.getEntry(i), false);
			}
			MatrixFunctions.vectorWriteCSV(folder, new ArrayRealVector(outA, false), true);
			//7
			for(int i = 0; i < y.getDimension(); i++) { 
				outA[i] = dist.dTF2(y.getEntry(i), mu.getEntry(i), sigma.getEntry(i),nu.getEntry(i), true);
			}
			MatrixFunctions.vectorWriteCSV(folder, new ArrayRealVector(outA, false), true);
			//8
			for(int i = 0; i < y.getDimension(); i++) { 
				outA[i] = dist.pTF2(y.getEntry(i), mu.getEntry(i), sigma.getEntry(i),nu.getEntry(i), true, true);
			}
			MatrixFunctions.vectorWriteCSV(folder, new ArrayRealVector(outA, false), true);
			//9
			for(int i = 0; i < y.getDimension(); i++) { 
				outA[i] = dist.pTF2(y.getEntry(i), mu.getEntry(i), sigma.getEntry(i),nu.getEntry(i), false, true);
			}
			MatrixFunctions.vectorWriteCSV(folder, new ArrayRealVector(outA, false), true);
			//10
			for(int i = 0; i < y.getDimension(); i++) { 
				outA[i] = dist.pTF2(y.getEntry(i), mu.getEntry(i), sigma.getEntry(i),nu.getEntry(i), true, false);
			}
			MatrixFunctions.vectorWriteCSV(folder, new ArrayRealVector(outA, false), true);
			//11
			
			for(int i = 0; i < y.getDimension(); i++) { 
				outA[i] = dist.pTF2(y.getEntry(i), mu.getEntry(i), sigma.getEntry(i),nu.getEntry(i), false, false);
			}
			MatrixFunctions.vectorWriteCSV(folder, new ArrayRealVector(outA, false), true);
			//12
			for(int i = 0; i < y.getDimension(); i++) { 
//				outA[i] = dist.qTF2(y.getEntry(i), mu.getEntry(i), sigma.getEntry(i),nu.getEntry(i), true, true);
			}
//			MatrixFunctions.vectorWriteCSV(folder, new ArrayRealVector(outA, false), true);
			//13
			for(int i = 0; i < y.getDimension(); i++) { 
//				outA[i] = dist.qTF2(y.getEntry(i), mu.getEntry(i), sigma.getEntry(i),nu.getEntry(i), false, true);
			}
//			MatrixFunctions.vectorWriteCSV(folder, new ArrayRealVector(outA, false), true);
			//14
			for(int i = 0; i < y.getDimension(); i++) { 
				outA[i] = dist.qTF2(y.getEntry(i), mu.getEntry(i), sigma.getEntry(i),nu.getEntry(i), true, false);
			}
			MatrixFunctions.vectorWriteCSV(folder, new ArrayRealVector(outA, false), true);
			//15
			for(int i = 0; i < y.getDimension(); i++) { 
				outA[i] = dist.qTF2(y.getEntry(i), mu.getEntry(i), sigma.getEntry(i),nu.getEntry(i), false, false);
			}
			MatrixFunctions.vectorWriteCSV(folder, new ArrayRealVector(outA, false), true);
			//16
			//for(int i = 0; i < y.getDimension(); i++) { 
			//	outA[i] = dist.dTF2(y.getEntry(i));
			//}
			//MatrixFunctions.vectorWriteCSV(folder, new ArrayRealVector(outA, false), true);
			//17
			//for(int i = 0; i < y.getDimension(); i++) { 
			//	outA[i] = dist.pTF2(y.getEntry(i));
			//}
			//MatrixFunctions.vectorWriteCSV(folder, new ArrayRealVector(outA, false), true);
			//18
			//for(int i = 0; i < y.getDimension(); i++) { 
			//	outA[i] = dist.qTF2(y.getEntry(i));
			//}
			//MatrixFunctions.vectorWriteCSV(folder, new ArrayRealVector(outA, false), true);
			//19
			out = dist.firstDerivative(DistributionSettings.NU, y);
			MatrixFunctions.vectorWriteCSV(folder, out, true);
			//20
			out = dist.secondDerivative(DistributionSettings.NU, y);
			MatrixFunctions.vectorWriteCSV(folder, out, true);
			//21
			out = dist.secondCrossDerivative(DistributionSettings.MU, DistributionSettings.NU,  y);
			MatrixFunctions.vectorWriteCSV(folder, out, true);
			//23
			out = dist.secondCrossDerivative(DistributionSettings.SIGMA, DistributionSettings.NU,  y);
			MatrixFunctions.vectorWriteCSV(folder, out, true);
			//
		
			System.out.println("Done !!!");
			
		}
		 
			 public static void main(final String[] args) {
				 new TestingDistributions(); 
			 }
	}