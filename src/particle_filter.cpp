/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;

// declare a random engine to be used across multiple and various method calls
//using std::normal_distribution;
//using std::uniform_int_distribution;
//using std::uniform_real_distribution;
using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
  std::default_random_engine gen;
  normal_distribution<double> dist_x(0, std[0]);
  normal_distribution<double> dist_y(0, std[1]);
  normal_distribution<double> dist_theta(0, std[2]);

  num_particles = 100;  // TODO: Set the number of particles
  for(int i = 0; i < num_particles; i++){
    Particle newParticle;
    newParticle.weight = 1.0;
    newParticle.id = i;
    newParticle.x = x;
    newParticle.y = y;
    newParticle.theta = theta;

    // gaussian sampling 
    newParticle.x += dist_x(gen);
    newParticle.y += dist_y(gen);
    newParticle.theta += dist_theta(gen);

    particles.push_back(newParticle);
  }
  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */
  std::default_random_engine gen;
  normal_distribution<double> dist_x(0, std_pos[0]);
  normal_distribution<double> dist_y(0, std_pos[1]);
  normal_distribution<double> dist_theta(0, std_pos[2]);

  for(int i = 0; i < num_particles; i++){
    if (fabs(yaw_rate) < 0.00001)
    {
      // this branch is necesarry to prevent devision by zero for 0 yaw_rate
      particles[i].x += velocity * delta_t * cos(particles[i].theta);
      particles[i].y += velocity * delta_t * sin(particles[i].theta);
    }
    else
    {
      particles[i].x += velocity / yaw_rate * (sin(particles[i].theta + yaw_rate * delta_t) - sin(particles[i].theta));
      particles[i].y += velocity / yaw_rate * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate*delta_t));
      particles[i].theta += yaw_rate * delta_t;
    }

    particles[i].x += dist_x(gen);
    particles[i].y += dist_y(gen);
    particles[i].theta += dist_theta(gen);
  }
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  /**
   * TODO: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */

  for(int i = 0;  i < observations.size(); i++){

    // get current observed landmark
    LandmarkObs currentObs = observations[i];
    // init minimum distance to maximum possible
    double min_dist = std::numeric_limits<double>::max();

    int map_id = -1;

    // check distances to all predicted positions
    for(int j = 0; j < predicted.size(); j++){
      LandmarkObs currentPred = predicted[j];

      double currentDist = dist(currentObs.x, currentObs.y, currentPred.x, currentPred.y);
      if(currentDist < min_dist){
        min_dist = currentDist;
        map_id = currentPred.id;
      }
    }

    // set the observations id to the nearest predicrted lanmarks id
    if (map_id != -1)
    {
      observations[i].id = map_id;
    }
    else
    {
      std::cout << "Data association failed!" << std::endl;
    }
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */

  for (int i = 0; i < num_particles; i++){
    double p_x = particles[i].x;
    double p_y = particles[i].y;
    double p_theta = particles[i].theta;

    vector<LandmarkObs> predictions;
    for (int j = 0; j < map_landmarks.landmark_list.size(); j++){
      double l_x = map_landmarks.landmark_list[j].x_f;
      double l_y = map_landmarks.landmark_list[j].y_f;
      int l_id = map_landmarks.landmark_list[j].id_i;

      // only landmarks within the sensor range have to be considered
      if (fabs(l_x - p_x) <= sensor_range && fabs(l_y - p_y) <= sensor_range){
        predictions.push_back(LandmarkObs{l_id, l_x, l_y});
      }
    }

    // create vector of observations in map coordinate system
    vector<LandmarkObs> os_map;
    for (int j = 0; j < observations.size(); j++){
      double x_map = cos(p_theta) * observations[j].x - sin(p_theta) * observations[j].y + p_x;
      double y_map = sin(p_theta) * observations[j].x + cos(p_theta) * observations[j].y + p_y;
      os_map.push_back(LandmarkObs{observations[j].id, x_map, y_map});
    }
    
    // perform data accociation step
    dataAssociation(predictions, os_map);
    particles[i].weight = 1.0;

    for (int j = 0; j < os_map.size(); j++){
      double x_map = os_map[j].x;
      double y_map = os_map[j].y;
      double x_pr, y_pr;

      int prediction_id = os_map[j].id;

      for(int k = 0; k < predictions.size(); k++){
        if(predictions[k].id == prediction_id){
          x_pr = predictions[k].x;
          y_pr = predictions[k].y;
        }
      }


      // calculate weights
      double s_x = std_landmark[0];
      double s_y = std_landmark[1];
      double gauss = (1/(2*M_PI*s_x*s_y)) * exp(-( pow(x_pr-x_map,2)/(2*pow(s_x, 2)) + (pow(y_pr-y_map,2)/(2*pow(s_y, 2)))));
    
      // apply updated weight
      particles[i].weight *= gauss;
    } 
  }
}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
  std::default_random_engine gen;
  vector<Particle> new_particles;

  vector<double> weights;
  for(int i = 0; i < num_particles; i++){
    weights.push_back(particles[i].weight);
  }

  // random starting index for "resampling wheel"
  uniform_int_distribution<int> uniintdist(0, num_particles-1);
  auto index = uniintdist(gen);

  // get max weight
  double max_weight = *max_element(weights.begin(), weights.end());

  // uniform random distribution [0.0, max_weight)
  uniform_real_distribution<double> unirealdist(0.0, max_weight);

  double beta = 0.0;
  // perform resampling
  for (int i = 0; i < num_particles; i++) {
    beta += unirealdist(gen) * 2.0;
    while (beta > weights[index]) {
      beta -= weights[index];
      index = (index + 1) % num_particles;
    }
    new_particles.push_back(particles[index]);
  }

  particles = new_particles;
}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}