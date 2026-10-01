

import rclpy
import numpy as np
import random

from math import pi, sin, cos, acos, atan2, sqrt, fmod, exp

# Grab the utilities
from hw5code.GeneratorNode      import GeneratorNode
from hw5code.TransformHelpers   import *
from hw5code.TrajectoryUtils    import *

# Grab the general fkin from HW5 P5.
from hw5code.KinematicChain     import KinematicChain

# Ball visualization
from rclpy.node                 import Node
from rclpy.qos                  import QoSProfile, DurabilityPolicy
from rclpy.time                 import Duration
from geometry_msgs.msg          import Point, Vector3, Quaternion
from std_msgs.msg               import ColorRGBA
from visualization_msgs.msg     import Marker
from visualization_msgs.msg     import MarkerArray

from demos.TransformHelpers     import *

RATE = 100

#
#   Atlas Joint Names
#
jointnames = ['l_leg_hpx', 'l_leg_hpy', 'l_leg_hpz',
              'l_leg_kny',
              'l_leg_akx', 'l_leg_aky',

              'r_leg_hpx', 'r_leg_hpy', 'r_leg_hpz',
              'r_leg_kny',
              'r_leg_akx', 'r_leg_aky',

              'back_bkx', 'back_bky', 'back_bkz',
              'neck_ry',

              'l_arm_elx', 'l_arm_ely',
              'l_arm_shx', 'l_arm_shz',
              'l_arm_wrx', 'l_arm_wry', 'l_arm_wry2',

              'r_arm_elx', 'r_arm_ely',
              'r_arm_shx', 'r_arm_shz',
              'r_arm_wrx', 'r_arm_wry', 'r_arm_wry2']

class Trajectory():
	def __init__(self,node):
		self.lchain = KinematicChain(node, 'pelvis', 'racket', ['back_bkz', 'back_bky',
		'back_bkx', 'l_arm_shz','l_arm_shx','l_arm_ely','l_arm_elx','l_arm_wry',
		'l_arm_wrx', 'l_arm_wry2'])
		self.rchain = KinematicChain(node, 'pelvis', 'r_hand',['back_bkz', 'back_bky',
		'back_bkx', 'r_arm_shz','r_arm_shx','r_arm_ely','r_arm_elx','r_arm_wry',
		'r_arm_wrx', 'r_arm_wry2'])
		self.tchain = KinematicChain(node, 'pelvis', 'head', ['back_bkz', 'back_bky',
		'back_bkx', 'neck_ry'])
		self.blchain = KinematicChain(node, 'pelvis', 'l_foot', ['l_leg_hpz', 'l_leg_hpx', 'l_leg_hpy',
		'l_leg_kny', 'l_leg_aky', 'l_leg_akx'])
		self.brchain = KinematicChain(node, 'pelvis', 'r_foot', ['r_leg_hpz', 'r_leg_hpx', 'r_leg_hpy', 'r_leg_kny', 'r_leg_aky', 'r_leg_akx'])
		#self.q0 = np.zeros((len(jointnames),1))
		self.q0 = np.array([0, 0.236, 0, -1.15, -0.212, 1.240, 0.815, -0.081, 0.6, -0.625, 0, -0.331, 0, 0.866, -0.587, 0.264, 1.150, 0.212, 1.24, -0.828, 0.081, -0.6, 0.625, 0, 0.337, 0, 0.866, -0.587, -0.264, 0]).reshape((-1,1))
		self.qd = np.array([0, 0.236, 0.663, -0.208, -0.212, 1.240, 0.815, -0.081, -0.372, -0.625, 0, -0.331, 0, 0.866, -0.587, 0.264, 1.150, 0.212, 1.24, -0.828, 0.081, -0.6, 0.625, 0, 0.337, 0, 0.866, -0.587, -0.264, 0]).reshape((-1,1))
		self.qd2 = np.array([0, 0.236, 0, -1.571, 0.25, 1.240, 0.815, -0.081, -0.372, -0.625, 0, -0.331, 0, 0.866, -0.587, 0.264, 1.150, 0.212, 1.24, -0.828, 0.081, -0.6, 0.625, 0, 0.337, 0, 0.866, -0.587, -0.264, 0]).reshape((-1,1))
		self.q0_la = self.q0[0:10] #np.array([0, 0, 0, 0, 0, 0, 0, -np.pi/2, 3*np.pi/2, np.pi/2]).reshape((-1,1))
		self.qd_la = self.qd[0:10] #np.array([np.pi/2, 0, 0, 0, np.pi/4, 0, -np.pi/4, -np.pi/2, 3*np.pi/2, np.pi/2]).reshape((-1,1))
		self.q0_ra = np.vstack((self.q0[0:3], self.q0[16:23])) #np.array([0, 0, 0, 0, 0, 0, 0, -np.pi/2, 3*np.pi/2, np.pi/2]).reshape((-1,1))
		self.qd_ra = np.vstack((self.qd[0:3], self.qd[16:23])) #np.array([np.pi/2, np.pi/2, 0, -np.pi/2, 0, 0, 0]).reshape((-1,1))
		self.q0_h = np.vstack((self.q0[0:3], self.q0[29:30])) #np.array([0]).reshape((-1,1))
		self.qd_h = np.vstack((self.qd[0:3], self.qd[29:30])) #np.array([np.pi/2]).reshape((-1,1))
		self.q0_rl = self.q0[10:16] #np.array([0,0,0,0,0,0]).reshape((-1,1))
		self.qd_rl = self.qd[10:16] #np.array([0,0,0,0,0,0]).reshape((-1,1))
		self.q0_ll = self.q0[23:29] #np.array([0,0,0,0,0,0]).reshape((-1,1))
		self.qd_ll = self.qd[23:29] #np.array([0,0,0,0,0,0]).reshape((-1,1))
		self.R0 = Reye()
		self.lam = 20
		self.q_la = self.q0_la
		self.q_ra = self.q0_ra
		self.q_ll = self.q0_ll
		self.q_rl = self.q0_rl
		self.q_n = self.q0_h
		#self.p0_lh = np.array([0.1281, 0.94115, 0.42859]).reshape((-1,1))
		self.p0_lh = self.lchain.fkin(self.q0_la)[0]
		#self.pd_lh = np.array([-0.468833, 0.41406, 0.64695635]).reshape((-1,1))
		#self.pd_lh = np.array([0, 1, 0.64695635]).reshape((-1,1))
		self.pd_lh = np.array([0, 1, 0.65]).reshape((-1,1))
		#self.p0_ll = np.array([0,0.1115,-0.862]).reshape((-1,1))
		self.p0_ll = self.blchain.fkin(self.q0_ll)[0]
		self.pd_ll = np.array([-1,0.1115,-0.862]).reshape((-1,1))
		#self.p0_rh = np.array([0.1281, -0.94115 , 0.42859]).reshape((-1,1))
		self.p0_rh = self.rchain.fkin(self.q0_ra)[0]
		self.pd_rh = np.array([.4, 0, 1.5]).reshape((-1,1))
		#self.p0_rl = np.array([0, -0.5 , 0]).reshape((-1,1))
		self.p0_rl = self.brchain.fkin(self.q0_rl)[0]
		self.pd_rl = np.array([0, -0.5 , 0]).reshape((-1,1))
		# this is wrong self.q = np.vstack((self.q_la, self.q_ll, self.q_ra, self.q_rl, self.q_n))
		self.pd_la_end = None
		self.pd_ra_end = None
		self.pd_rl_end = None
		self.Rd_la = None
		self.ball_exists = False
		
		self.qd2_lh = self.qd2[0:10]
		#self.pd2_lh = self.lchain.fkin(self.qd2_lh)[0]
		self.pd2_lh = np.array([1, -4, 1.1]).reshape((-1,1))
		self.pd3_lh = np.array([1, -4, 1.1]).reshape((-1,1))
		
		self.q_start = self.q0[0:10]
		self.ball_hit = False

		##########

		# Prepare the publisher (latching for new subscribers).
		quality = QoSProfile(durability=DurabilityPolicy.TRANSIENT_LOCAL,
		                     depth=1)
		self.pub = node.create_publisher(
		    MarkerArray, '/visualization_marker_array', quality)

		# Initialize the ball position, velocity, set the acceleration.
		self.radius = 0.05

		(pi, pf) = self.find_initial_ball()
		self.p = pi
		self.pf = pf

		self.v = np.array([((pf[0,0] - pi[0,0])/2)*(1/1.5), 0, 3.0]).reshape((3,1))
		print("pf", self.pf)
		self.a = np.array([0.0, 0.0, -9.81]).reshape((3,1))

		self.rac_bounced = False
		self.ball_on_rac = 0
		self.pball_close_to_rac = False

		# Create the sphere marker.
		diam        = 2 * self.radius
		self.marker = Marker()
		self.marker.header.frame_id  = "l_foot"
		self.marker.header.stamp     = node.get_clock().now().to_msg()
		self.marker.action           = Marker.ADD
		self.marker.ns               = "point"
		self.marker.id               = 1
		self.marker.type             = Marker.SPHERE
		self.marker.pose.orientation = Quaternion()
		self.marker.pose.position    = Point_from_p(self.p)
		self.marker.scale            = Vector3(x = diam, y = diam, z = diam)
		self.marker.color            = ColorRGBA(r=0.0, g=1.0, b=0.0, a=0.8)

		# Create the marker array message.
		self.mark = MarkerArray()
		self.mark.markers.append(self.marker)

		# Set up the timing so (t=0) will occur in the first update
		# cycle (dt) from now.
		self.dt    = 1.0 / float(RATE)
		self.t     = -self.dt
		self.start = node.get_clock().now() + Duration(seconds=self.dt)

		# Create a timer to keep calling up  date().
		node.create_timer(self.dt, node.update)
		node.get_logger().info("Running with dt of %f seconds (%fHz)" %
		                       (self.dt, RATE))

		self.vbounce = self.v 
	def shutdown(self):
		self.destroy_node()

	def jointnames(self):
		return ['back_bkz', 'back_bky',
		'back_bkx', 'l_arm_shz','l_arm_shx','l_arm_ely','l_arm_elx','l_arm_wry',
		'l_arm_wrx', 'l_arm_wry2', 'l_leg_hpz', 'l_leg_hpx', 'l_leg_hpy', 'l_leg_kny', 'l_leg_aky', 'l_leg_akx', 'r_arm_shz','r_arm_shx','r_arm_ely','r_arm_elx','r_arm_wry',
		'r_arm_wrx', 'r_arm_wry2', 'r_leg_hpz', 'r_leg_hpx', 'r_leg_hpy', 'r_leg_kny', 'r_leg_aky', 'r_leg_akx', 'neck_ry']
	
	def find_initial_ball(self):
		t = 1.5
		pfx = 0
		#pfy = random.uniform(0.5,1.5)
		pfy = random.uniform(0,2)
		#pfz = random.uniform(0.15, 1.15)
		pfz = random.uniform(-.35, 1.65)
		pf = np.array([pfx, pfy, pfz]).reshape((-1,1))
		pi = np.array([4,1.0,pfz]).reshape((-1,1))
		print("pf: ", pf)
		return (pi, pf)
	    
	def close_to_rac(self, prac, Rrac): # wrt l_foot = world
		pball_wrt_rac = Rrac.T @ (self.p - prac)
		vball_wrt_rac = Rrac.T @ (self.v)
		return (pball_wrt_rac, vball_wrt_rac)
	
	def evaluate(self, t, dt):
		t = t % 4.5
		if t <= 3:
			# LEFT ARM
			(pd_la,vd_la) = spline(t, 3.0, self.p0_lh, self.pf, np.array([-1, 0, 0]).reshape((-1,1)),np.array([2, 0, 0]).reshape((-1,1)))
			(alpha, alphadot) = goto(t, 3.0, 0, 0)
			Rd_la = Roty(alpha)
			wd_la = ey()*alphadot
			# RIGHT ARM
			(pd_ra,vd_ra) = goto(t, 3.0, self.p0_rh, self.pd_rh)
			(alpha, alphadot) = goto(t, 3.0, 0, 0)
			Rd_ra = Rotz(alpha)
			wd_ra = ez()*alphadot
			# RIGHT LEG
			(pd_rl,vd_rl) = goto(t, 3.0, self.p0_rl, self.pd_rl)
			(alpha, alphadot) = goto(t, 3.0, 0, 0)
			Rd_rl = Rotz(alpha)
			wd_rl = ez()*alphadot
			
			self.pd_la_end = pd_la
			self.pd_ra_end = pd_ra
			self.pd_rl_end = pd_rl
			self.Rd_la = Rd_la
		#else:
			#if not(self.pball_close_to_rac) and not(self.ball_hit):
				#print("false")
				#t = 0
				#pd_la = self.pd_la_end
				#vd_la = np.zeros((3,1))
				#Rd_la = Reye()
				#wd_la = np.zeros((3,1))

				#pd_ra = self.pd_ra_end
				#vd_ra = np.zeros((3,1))
				#Rd_ra = Reye()
				#wd_ra = np.zeros((3,1))

				#pd_rl = self.pd_rl_end
				#vd_rl = np.zeros((3,1))
				#Rd_rl = Reye()
				#wd_rl = np.zeros((3,1))								
			#elif self.ball_hit:
				#print("true")
				#t = t%3
				#print(t)
		elif t >= 3.0 and t < 4.5:
				(qlast_la, qdot_la) = goto(t-3.0, 1.5, self.q_start, self.q0_la )#self.qd2_lh)
				qdot = np.vstack((qdot_la, np.zeros((20,1))))
				(p_la, R_la, Jv_la, Jw_la) = self.lchain.fkin(qlast_la)
				#(alpha, alphadot) = goto(t, .5, 0, 0)
				#Rd_la = Roty(alpha)
				#wd_la = ey()*alphadot
			
				(pd_ra,vd_ra) = goto(t-3.0, 1.5, self.p0_rh, self.pd_rh)
				(alpha, alphadot) = goto(t, 1.5, 0, 0)
				Rd_ra = Rotz(alpha)
				wd_ra = ez()*alphadot
			
				(pd_rl,vd_rl) = goto(t-3.0, 1.5, self.p0_rl, self.pd_rl)
				(alpha, alphadot) = goto(t, 1.5, 0, 0)
				Rd_rl = Rotz(alpha)
				wd_rl = ez()*alphadot
				
		#elif t >= 3.5 and t <= 4.5:
				#(qlast_la, qdot_la) = goto(t-3.5, 1.0, self.qd2_lh, self.q0_la)
				#(p_la, R_la, Jv_la, Jw_la) = self.lchain.fkin(qlast_la)
				#qdot = np.vstack((qdot_la, np.zeros((20,1))))
					
				
				#pd_la = self.pd_la_end
				#vd_la = np.zeros((3,1))
				#Rd_la = Reye()
				#wd_la = np.zeros((3,1))

				#pd_ra = self.pd_ra_end
				#vd_ra = np.zeros((3,1))
				#Rd_ra = Reye()
				#wd_ra = np.zeros((3,1))

				#pd_rl = self.pd_rl_end
				#vd_rl = np.zeros((3,1))
				#Rd_rl = Reye()
				#wd_rl = np.zeros((3,1))	
			# (p, v, Jv, Jw) = self.rchain.fkin(self.q0_ra)
			# return None
		if not(self.pball_close_to_rac) and self.ball_hit:
			self.ball_hit = True
		if t < 3: 
			# Left leg
			qlast_ll = self.q_ll
			(p_ll, R_ll, Jv_ll, Jw_ll) = self.blchain.fkin(qlast_ll)
			J_zeros_ll_1 = np.zeros((3,10))
			J_zeros_ll_2 = np.zeros((3,14))
			J_lp_v = np.hstack((J_zeros_ll_1, Jv_ll, J_zeros_ll_2))
			J_lp_w = np.hstack((J_zeros_ll_1, Jw_ll, J_zeros_ll_2))
			
			# Left arm
			qlast_la = self.q_la
			(p_la, R_la, Jv_la, Jw_la) = self.lchain.fkin(qlast_la)
			p_la_rel = R_ll.T @ (p_la - p_ll)
			error_trans_la = ep(pd_la, p_la_rel)
			R_la_rel = R_ll.T @ R_la
			error_rot_la = eR(Rd_la, R_la_rel)
			error_la = np.vstack((error_trans_la, error_rot_la))
			J_zeros_la = np.zeros((3, 20))
			J_lh_p_v = np.hstack((Jv_la, J_zeros_la))
			J_lh_l_v = R_ll.T @ (J_lh_p_v - J_lp_v)
			J_lh_p_w = np.hstack((Jw_la, J_zeros_la))
			J_lh_l_w = R_ll.T @ (J_lh_p_w - J_lp_w)
			J_la = np.vstack((J_lh_l_v, J_lh_l_w))
			v_la = np.vstack((vd_la,wd_la))
			
			# Right arm
			qlast_ra = self.q_ra
			(p_ra, R_ra, Jv_ra, Jw_ra) = self.rchain.fkin(qlast_ra)
			p_ra_rel = R_ll.T @ (p_ra - p_ll)
			error_trans_ra = ep(pd_ra, p_ra_rel)
			error_rot_ra = eR(Rd_ra, R_ra)
			error_ra = np.vstack((error_trans_ra, error_rot_ra))
			J_torso_ra_v = Jv_ra[:, 0:3]
			J_torso_ra_w = Jw_ra[:, 0:3]
			Jw_ra_ra = Jw_ra[:, 3:10]
			Jv_ra_ra = Jv_ra[:, 3:10]
			J_zeros_ra_1 = np.zeros((3, 13))
			J_zeros_ra_2 = np.zeros((3, 7))
			J_rh_p_v = np.hstack((J_torso_ra_v, J_zeros_ra_1, Jv_ra_ra, J_zeros_ra_2))
			J_rh_l_v = R_ll.T @ (J_rh_p_v - J_lp_v)
			J_rh_p_w = np.hstack((J_torso_ra_w, J_zeros_ra_1, Jw_ra_ra, J_zeros_ra_2))
			J_rh_l_w = R_ll.T @ (J_rh_p_w - J_lp_w)	
			J_ra = np.vstack((J_rh_l_v, J_rh_l_w))
			v_ra = np.vstack((vd_ra, wd_ra))
			
			# Right leg
			qlast_rl = self.q_rl
			(p_rl, R_rl, Jv_rl, Jw_rl) = self.brchain.fkin(qlast_rl)
			p_rl_rel = R_ll.T @ (p_rl - p_ll)
			error_trans_rl = ep(pd_rl, p_rl_rel)
			error_rot_rl = eR(Rd_rl, R_rl)
			error_rl = np.vstack((error_trans_rl, error_rot_rl))
			J_zeros_rl_1 = np.zeros((3,23))
			J_zeros_rl_2 = np.zeros((3,1))
			J_rf_p_v = np.hstack((J_zeros_rl_1, Jv_rl, J_zeros_rl_2))
			J_rf_l_v = R_ll.T @ (J_rf_p_v - J_lp_v)
			J_rf_p_w = np.hstack((J_zeros_rl_1, Jw_rl, J_zeros_rl_2))
			J_rf_l_w = R_ll.T @ (J_rf_p_w - J_lp_w)
			J_rl = np.vstack((J_rf_l_v, J_rf_l_w))
			v_rl = np.vstack((vd_rl, wd_rl))
		
			#qgoal_la = np.array([0.663, 0, 0, 0.785, -0.263, 1.817, 1.134, 0, 0, -0.5]).reshape((-1,1))		
			#qgoal_ra = np.array([0.663,0,0, 1.571, 0, 1.562, -0.751, 0, 0, 0]).reshape((-1,1))
			#qgoal = np.vstack((qgoal_la, np.zeros((6,1)), qgoal_ra[3:10], np.zeros((7,1))))
			qgoal = self.qd
			qcur = np.vstack((self.q_la, np.zeros((6,1)), self.q_ra[3:10], np.zeros((7,1))))
			qdot_sec = 10 * (qgoal - qcur)
			
			# Final q and qdot measurements
			#J = np.vstack((J_la, J_ra))
			#J = J_la
			J = np.vstack((J_la, 0*J_ra, 0*J_rl))
			#J = J_ra
			#v = np.vstack((v_la, v_ra))
			#v = v_la
			v = np.vstack((v_la, 0*v_ra, 0*v_rl))
			#v = v_ra
			#error = np.vstack((error_la, error_ra))
			#error = error_la
			error = np.vstack((error_la, 0*error_ra, 0*error_rl))
			#error = error_ra
			qdot = np.linalg.pinv(J) @ (v + self.lam*error) + (np.eye(30) - np.linalg.pinv(J) @ J) @ qdot_sec 
			
			
			qdot[1,0] = 0
			qdot[2,0] = 0
			if t > 3: 
				qdot[0,0] = 0
				qdot[1,0] = 0
				qdot[2,0] = 0
				#qdot[3,0] = 0
				#qdot[4,0] = 0
				qdot[5,0] = 0
				#qdot[6,0] = 0
				qdot[7,0] = 0
				qdot[8,0] = 0
				qdot[9,0] = 0
			qdot[10,0] = 0
			qdot[11,0] = 0
			qdot[12,0] = 0
			qdot[13,0] = 0
			qdot[14,0] = 0
			qdot[15,0] = 0
			qdot[23,0] = 0
			qdot[24,0] = 0
			qdot[25,0] = 0
			qdot[26,0] = 0
			qdot[27,0] = 0
			qdot[28,0] = 0
			qdot[29,0] = 0
			
			self.q_start = self.q_la

		qlast_ll = self.q_ll
		qlast_ra = self.q_ra
		qlast_rl = self.q_rl
		qlast = np.vstack((qlast_la, qlast_ll, qlast_ra[3:10], qlast_rl, np.zeros((1,1))))
		q = qlast + dt*qdot
		self.q_la = q[0:10]
		self.q_ll = q[10:16]
		self.q_ra = np.vstack((q[0:3], q[16:23]))
		self.q_rl = q[23:29]	
		
		
		if t < 5:
		# To avoid any time jitter enforce a constant time step and
		# integrate to get the current time.
			self.t += self.dt
		
		# Integrate the velocity, then the position.
			self.v += self.dt * self.a
			self.p += self.dt * self.v
		
			(pball_wrt_rac, vball_wrt_rac) = self.close_to_rac(p_la, R_la)
			close_pball_x = abs(pball_wrt_rac[0, 0]) < 0.6
			close_pball_y = abs(pball_wrt_rac[1, 0]) < 0.6
			close_pball_z = abs(pball_wrt_rac[2, 0]) < 0.6

			self.pball_close_to_rac = (close_pball_x and close_pball_y and close_pball_z)
		# bounce off ground
			if self.p[2,0] < self.radius:
				self.p[2,0] = self.radius + (self.radius - self.p[2,0])
				self.v[0,0] *= 1.0
				self.v[1,0] *= 1.0
				self.v[2,0] *= -1.0
				print('bounce ground')
				self.rac_bounced = False
				self.marker.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=0.8)
		
		
		# bounce off rac
			elif (close_pball_x and close_pball_y and close_pball_z) and self.rac_bounced == False:
				print('close to rac!!!!!!!!!!!!')
			
				at_pball_x = abs(pball_wrt_rac[0, 0]) < 0.1
				at_pball_y = abs(pball_wrt_rac[1, 0]) < 0.1
				at_pball_z = abs(pball_wrt_rac[2, 0]) < 0.1

			# move towards center
				if t <= 3.0:#not(self.p[0, 0] == p_la_rel[0, 0] and self.p[1, 0] == p_la_rel[1, 0] and self.p[2, 0] == p_la_rel[2, 0]): #(at_pball_x and at_pball_y and at_pball_z) and t < 3.1:
					attractx = self.p[0, 0] - p_la[0, 0]
					attracty = self.p[1, 0] - p_la[1, 0]
					attractz = self.p[2, 0] - p_la[2, 0]
					self.vbounce = self.v
					self.v = np.array((attractx, attracty, attractz)).reshape((-1, 1))
					self.p = p_la_rel
					print ("GREEN")

			# # hold ball at center
			# elif (at_pball_x and at_pball_y and at_pball_z) and self.ball_on_rac < 5:
			# 	self.p = np.array((p_la[0, 0], p_la[1, 0], p_la[2, 0])).reshape((-1, 1))
			# 	self.v = np.array((v_la[0, 0], v_la[1, 0], v_la[2, 0])).reshape((-1, 1))
			# 	self.ball_on_rac += 1
			
				if t > 3.0:
					print('rac bounce!!!!!!!!!!!')
					self.ball_on_rac = 0
					self.ball_hit = True
					self.rac_bounced = True
					self.v_saved = False
					self.pball_close_to_rac = True
					print("RED")

					#self.p = p_la_rel
				# self.p = pball_wrt_rac
					#self.v = np.array((-1*vball_wrt_rac[0,0], -1*vball_wrt_rac[1,0], vball_wrt_rac[2,0])).reshape((-1,1))
					self.v = np.array((1.3 ,0,3.0)).reshape((-1,1))
					self.marker.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=0.8)
				#self.p = np.array([5, 0.0, self.radius]).reshape((3,1))

		# Update the ID number to create a new ball and leave the
		# previous balls where they are.
			if self.p[1, 0] > 2.0 or self.p[1, 0] < -2.0 or self.p[0, 0] > 6.0 or self.p[0, 0] < -2.0 or t > 4.25:
				(pi, pf) = self.find_initial_ball()
				self.ball_exists = True
				self.p = pi
				self.pf = pf
				self.v = np.array([((pf[0,0] - pi[0,0])/2)*(1/1.5), 0, 3.0]).reshape((3,1))
			#if self.t > 4.5:
				#(pi, pf) = self.find_initial_ball()
				#self.p = pi
				#self.pf = pf
				#self.v = np.array([((pf[0,0] - pi[0,0])/2)*(1/2), 0, 5]).reshape((3,1))
				#self.ball_exists = False
		
		#####################
		# self.marker.id += 1
		#####################

		# Update the message and publish.
		now = self.start + Duration(seconds=self.t)
		self.marker.header.stamp  = now.to_msg()
		self.marker.pose.position = Point_from_p(self.p)
		self.pub.publish(self.mark)

		return (q.flatten().tolist(), qdot.flatten().tolist())
#
#  Main Code
#
def main(args=None):
    # Initialize ROS.
    rclpy.init(args=args)

    # Initialize the generator node for 100Hz udpates, using the above
    # Trajectory class.
    generator = GeneratorNode('generator', RATE, Trajectory)

    # Spin, meaning keep running (taking care of the timer callbacks
    # and message passing), until interrupted or the trajectory ends.
    generator.spin()
    
    # Shutdown the node and ROS.
    generator.shutdown()

    rclpy.shutdown()

if __name__ == "__main__":
    main()
