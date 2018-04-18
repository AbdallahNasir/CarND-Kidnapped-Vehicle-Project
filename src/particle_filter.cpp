/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"
#include "helper_functions.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
    // TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
    //   x, y, theta and their uncertainties from GPS) and all weights to 1. 
    // Add random Gaussian noise to each particle.
    // NOTE: Consult particle_filter.h for more information about this method (and others in this file).
    num_particles = 100;

    default_random_engine gen;
    normal_distribution<double> dist_x(x, std[0]);
    normal_distribution<double> dist_y(y, std[1]);
    normal_distribution<double> dist_theta(theta, std[2]);

    weights.resize(num_particles);
    particles.resize(num_particles);
    for (int i = 0; i < num_particles; i++) {
        Particle newParticle = Particle();
        newParticle.id = i;
        newParticle.x = dist_x(gen);
        newParticle.y = dist_y(gen);
        newParticle.theta = dist_theta(gen);

        newParticle.weight = 1;
        weights[i] = 1;
        particles[i] = newParticle;
    }
    is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
    // TODO: Add measurements to each particle and add random Gaussian noise.
    // NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
    //  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
    //  http://www.cplusplus.com/reference/random/default_random_engine/

    default_random_engine gen;

    for (uint i = 0; i < particles.size(); i++) {
        Particle dp = particles[i];
        double xnew;
        double ynew;
        double thetanew;

        if (fabs(yaw_rate) <= 0.00001) {
            xnew = dp.x + (velocity * delta_t) * cos(dp.theta);
            ynew = dp.y + (velocity * delta_t) * sin(dp.theta);
            thetanew = dp.theta;
        } else {
            xnew = dp.x + velocity / yaw_rate * (sin(dp.theta + yaw_rate * delta_t) - sin(dp.theta));
            ynew = dp.y + velocity / yaw_rate * (cos(dp.theta) - cos(dp.theta + yaw_rate * delta_t));
            thetanew = dp.theta + yaw_rate * delta_t;

        }

        normal_distribution<double> dist_x(xnew, std_pos[0]);
        normal_distribution<double> dist_y(ynew, std_pos[1]);
        normal_distribution<double> dist_theta(thetanew, std_pos[2]);

        particles[i].x = dist_x(gen);
        particles[i].y = dist_y(gen);
        particles[i].theta = dist_theta(gen);

    }
}

void ParticleFilter::dataAssociation(std::vector<Map::single_landmark_s> reaLandmarks, std::vector<LandmarkObs>& observations) {
    // TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
    //   observed measurement to this particular landmark.
    // NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
    //   implement this method and use it as a helper during the updateWeights phase.

    for (uint i = 0; i < observations.size(); i++) {
        double mindist = 9999999999999999;
        for (uint j = 0; j < reaLandmarks.size(); j++) {
            double tempDist = dist(observations[i].x, observations[i].y, reaLandmarks[j].x_f, reaLandmarks[j].y_f);
            if (tempDist < mindist) {
                mindist = tempDist;
                observations[i].id = j;
            }
        }
    }

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
        const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
    // TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
    //   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
    // NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
    //   according to the MAP'S coordinate system. You will need to transform between the two systems.
    //   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
    //   The following is a good resource for the theory:
    //   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
    //   and the following is a good resource for the actual equation to implement (look at equation 
    //   3.33
    //   http://planning.cs.uiuc.edu/node99.html


    for (uint i = 0; i < particles.size(); i++) {
        Particle dp = particles[i];
        std::vector<LandmarkObs> map_observations;
        map_observations.resize(observations.size());
        for (uint j = 0; j < observations.size(); j++) {
            LandmarkObs d_observation = observations[j];
            LandmarkObs map_observation = LandmarkObs();

            map_observation.x = dp.x + (cos(dp.theta) * d_observation.x) - (sin(dp.theta) * d_observation.y);
            map_observation.y = dp.y + (sin(dp.theta) * d_observation.x) + (cos(dp.theta) * d_observation.y);

            map_observations[j] = map_observation;
        }

        dataAssociation(map_landmarks.landmark_list, map_observations);
        particles[i].associations.clear();
        particles[i].sense_x.clear();
        particles[i].sense_y.clear();
        double sig_x = std_landmark[0];
        double sig_y = std_landmark[1];
        double weight = 1.0;
        for (uint j = 0; j < map_observations.size(); j++) {


            LandmarkObs d_observation = map_observations[j];
            double x_obs = d_observation.x;
            double y_obs = d_observation.y;

            particles[i].associations.push_back(map_landmarks.landmark_list[d_observation.id].id_i);
            particles[i].sense_x.push_back(map_landmarks.landmark_list[d_observation.id].x_f);
            particles[i].sense_y.push_back(map_landmarks.landmark_list[d_observation.id].y_f);

            double mu_x = map_landmarks.landmark_list[d_observation.id].x_f;
            double mu_y = map_landmarks.landmark_list[d_observation.id].y_f;
            double gauss_norm = 1.0 / (2.0 * M_PI * sig_x * sig_y);
            double exponent = pow((x_obs - mu_x), 2.0) / (2.0 * pow(sig_x, 2.0)) + pow((y_obs - mu_y), 2.0) / (2.0 * pow(sig_y, 2.0));
            weight = weight * (gauss_norm * exp(-1 * exponent));
        }
        particles[i].weight = weight;
        weights[i] = weight;
    }
}

void ParticleFilter::resample() {
    // TODO: Resample particles with replacement with probability proportional to their weight. 
    // NOTE: You may find std::discrete_distribution helpful here.
    //   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution


    default_random_engine gen;

    double wsum = 0.0;
    for (int i = 0; i < num_particles; i++) {
        wsum += particles[i].weight;
    }

    for (int i = 0; i < num_particles; i++) {
        weights[i] = particles[i].weight / wsum;
    }


    std::discrete_distribution<uint> d(weights.begin(), weights.end());

    std::vector<Particle> newparticles;
    newparticles.resize(num_particles);
    for (int i = 0; i < num_particles; i++) {
        uint id = d(gen);
        Particle newOne = particles[id];
        newparticles[i] = newOne;
    }
    particles = newparticles;
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations,
        const std::vector<double>& sense_x, const std::vector<double>& sense_y) {
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations = associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
    return particle;
}

string ParticleFilter::getAssociations(Particle best) {
    vector<int> v = best.associations;
    stringstream ss;
    copy(v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length() - 1); // get rid of the trailing space
    return s;
}

string ParticleFilter::getSenseX(Particle best) {
    vector<double> v = best.sense_x;
    stringstream ss;
    copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length() - 1); // get rid of the trailing space
    return s;
}

string ParticleFilter::getSenseY(Particle best) {
    vector<double> v = best.sense_y;
    stringstream ss;
    copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length() - 1); // get rid of the trailing space
    return s;
}
