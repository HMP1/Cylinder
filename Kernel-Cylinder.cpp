__kernel void CylinderKernel(__global const float *qx, global const float *qy, __global float *_ptvalue, const float sld_solv,
const float sld_cyl, const float m_max, const float m_max_solv, const float rr, const float h, const float scale,
const float radius_weight, const float length_weight, const float theta_weight, const float phi_weight, const float cyl_theta,
const float cyl_phi, const float background, const int count, const int size)
{
	// qq is the q-value for the calculation (1/A)
	// rr is the radius of the cylinder (A)
	// h is the -LENGTH of the cylinder = L (A)
    int i = get_global_id(0);

    if(i < count)
    {
        float qq = sqrt(qx[i]*qx[i]+qy[i]*qy[i]);

        float pi = 4.0*atan(1.0);
        float theta = cyl_theta*pi/180.0;
        float phi = cyl_phi*pi/180.0;

        float cyl_x = cos(theta)*cos(phi);
        float cyl_y = sin(theta);
        float cos_val = cyl_x*(qx[i]/qq) + cyl_y*(qy[i]/qq);

        float alpha = acos(cos_val);
        if(alpha == 0.0){
            alpha = 1.0e-26;
        }
        float besarg = qq*rr*sin(alpha);
        float siarg = qq*h/2*cos(alpha);

	    float xx=0.0; float y=0.0; float bj=0.0; float ans1=0.0; float ans2=0.0; float z=0.0; float answer=0.0;
	    float contrast=0.0; float form=0.0; float be=0.0; float si=0.0;

        float ax = fabs(besarg);

        if(ax < 8.0) {
            y=besarg*besarg;
            ans1=besarg*(72362614232.0+y*(-7895059235.0+y*(242396853.1+y*(-2972611.439+y*(15704.48260+y*(-30.16036606))))));
            ans2=144725228442.0+y*(2300535178.0+y*(18583304.74+y*(99447.43394+y*(376.9991397+y*1.0))));
            bj=ans1/ans2;
        }
        else{
            z=8.0/ax;
            y=z*z;
            xx=ax - 2.356194491;
            ans1=1.0+y*(0.183105e-2+y*(-0.3516396496e-4+y*(0.2457520174e-5+y*(-0.240337019e-6))));
            ans2=0.04687499995+y*(-0.2002690873e-3+y*(0.8449199096e-5+y*(-0.88228987e-6+y*0.105787412e-6)));
            bj=sqrt(0.636619772/ax)*(cos(xx)*ans1-z*sin(xx)*ans2);

            if(besarg < 0.0){bj*=-1;}
        }

        float t1 = sin(siarg)*sin(siarg);
        float t2 = bj*bj*4.0*sin(alpha);
        float b1 = siarg*siarg;
        float d1 = qq*rr*sin(alpha);
        float b2 = d1*d1;

        if (besarg == 0.0) {be = sin(alpha);}
        else {be = t2/b2;}
        if(siarg == 0.0) {si = 1.0;}
        else{si = t1/b1;}

        form = be*si/sin(alpha);

        if (m_max < 1.0e-32 && m_max_solv < 1.0e-32){
            contrast = sld_cyl - sld_solv;
            answer = contrast*contrast*form;
        }
/*
	else{
		double qx = q_x;
		double qy = q_y;
		double s_theta = pars->Up_theta;
		double m_phi = pars->M_phi_cyl;
		double m_theta = pars->M_theta_cyl;
		double m_phi_solv = pars->M_phi_solv;
		double m_theta_solv = pars->M_theta_solv;
		double in_spin = pars->Up_frac_i;
		double out_spin = pars->Up_frac_f;
		polar_sld p_sld;
		polar_sld p_sld_solv;
		p_sld = cal_msld(1, qx, qy, sld_cyl, m_max, m_theta, m_phi,
			 			in_spin, out_spin, s_theta);
		p_sld_solv = cal_msld(1, qx, qy, sld_solv, m_max_solv, m_theta_solv, m_phi_solv,
			 			in_spin, out_spin, s_theta);
		//up_up
		if (in_spin > 0.0 && out_spin > 0.0){
			answer += ((p_sld.uu- p_sld_solv.uu) * (p_sld.uu- p_sld_solv.uu) * form);
			}
		//down_down
		if (in_spin < 1.0 && out_spin < 1.0){
			answer += ((p_sld.dd - p_sld_solv.dd) * (p_sld.dd - p_sld_solv.dd) * form);
			}
		//up_down
		if (in_spin > 0.0 && out_spin < 1.0){
			answer += ((p_sld.re_ud - p_sld_solv.re_ud) * (p_sld.re_ud - p_sld_solv.re_ud) * form);
			answer += ((p_sld.im_ud - p_sld_solv.im_ud) * (p_sld.im_ud - p_sld_solv.im_ud) * form);
			}
		//down_up
		if (in_spin < 1.0 && out_spin > 0.0){
			answer += ((p_sld.re_du - p_sld_solv.re_du) * (p_sld.re_du - p_sld_solv.re_du) * form);
			answer += ((p_sld.im_du - p_sld_solv.im_du) * (p_sld.im_du - p_sld_solv.im_du) * form);
			}
	}
*/
    //normalize by cylinder volume
    //NOTE that for this (Fournet) definition of the integral, one must MULTIPLY by Vcyl
    answer *= acos(-1.0)*rr*rr*h*1.0e8*scale;
    //answer += background;

    //Un-normalize by volume
    _ptvalue[i] = radius_weight*length_weight*theta_weight*phi_weight*answer*pow(rr,2)*h;
	if (size>1) {
	    _ptvalue[i] *= fabs(cos(cyl_theta*pi/180.0));
	}
}
}
