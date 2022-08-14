#include <mitsuba/core/properties.h>
#include <mitsuba/core/warp.h>
#include <mitsuba/core/spectrum.h>
#include <mitsuba/render/emitter.h>
#include <mitsuba/render/medium.h>
#include <mitsuba/render/shape.h>
#include <mitsuba/render/texture.h>
#include <mitsuba/render/ior.h>

NAMESPACE_BEGIN(mitsuba)

template <typename Float, typename Spectrum>
class DisplayEmitter final : public Emitter<Float, Spectrum> {
public:
    MI_IMPORT_BASE(Emitter, m_flags, m_shape, m_medium)
    MI_IMPORT_TYPES(Scene, Shape, Texture)

    DisplayEmitter(const Properties &props) : Base(props) {

        ScalarFloat int_ior = lookup_ior(props, "int_ior", "bk7");
        ScalarFloat ext_ior = lookup_ior(props, "ext_ior", "air"); // water

        // n2 / n1 here!!
        // From fresnel.h:
        // > A value greater than 1.0 case means that the surface normal
        // > points into the region of lower density.
        // Beware, usually we use eta = n1/n2, but not here!
        m_eta = int_ior / ext_ior;

        m_thickness = props.get<ScalarFloat>("thickness", 0);

        m_polarization_dir = ScalarVector2f(props.get<ScalarFloat>("polarization_x", 1.0f), props.get<ScalarFloat>("polarization_y", -1.0f));
        m_polarization_dir = dr::normalize(m_polarization_dir);

        m_emission_coeffs[0] = props.get<ScalarFloat>("emission_coeff_0", 1.0f);
        m_emission_coeffs[1] = props.get<ScalarFloat>("emission_coeff_1", 0.0f);
        m_emission_coeffs[2] = props.get<ScalarFloat>("emission_coeff_2", 0.0f);
        m_emission_coeffs[3] = props.get<ScalarFloat>("emission_coeff_3", 0.0f);
        m_emission_coeffs[4] = props.get<ScalarFloat>("emission_coeff_4", 0.0f);
        m_emission_coeffs[5] = props.get<ScalarFloat>("emission_coeff_5", 0.0f);

        m_radiance = props.texture<Texture>("radiance", Texture::D65(1.f));

        m_flags = +EmitterFlags::Surface;
        if (m_radiance->is_spatially_varying())
            m_flags |= +EmitterFlags::SpatiallyVarying;
    }


    Spectrum compute_display_emission_profile(Float cos_theta_t) const
    {
        Spectrum result = 0;
        for (size_t i = 0; i < m_emission_coeffs.size(); ++i)
        {
            result = dr::fmadd(result, cos_theta_t, m_emission_coeffs[m_emission_coeffs.size()-1-i]);
        }
        return result;
    }

    Spectrum compute_display_glass_transmittance(Vector3f wi) const
    {
        // Evaluate the Fresnel equations for unpolarized illumination
        Float cos_theta_i = Frame3f::cos_theta(wi);
        auto [ a_s, a_p, cos_theta_t, eta_it, eta_ti ] = fresnel_polarized(cos_theta_i, m_eta);
        Float R_s = dr::squared_norm(a_s);
        Float R_p = dr::squared_norm(a_p);
        Float T_s = 1 - R_s;
        Float T_p = 1 - R_p;

        // TODO handle parallel_dir close to 0!
        Vector2f parallel_dir = dr::normalize(dr::head<2>(wi));

        // displayToTransmissionPlaneMat = torch.stack([px, py, -py, px], dim=-1).view(*parallel_dir.shape[:-1], 2, 2)
        // J = fct.matvecmul(displayToTransmissionPlaneMat, args.displayPolarization)
        Float J_p = parallel_dir.x()*m_polarization_dir.x() + parallel_dir.y()*m_polarization_dir.y(); // Parallel
        Float J_s = parallel_dir.x()*m_polarization_dir.y() - parallel_dir.y()*m_polarization_dir.x(); // Senkrecht

        // Transmittance
        Float T = (dr::sqr(J_s)*T_s + dr::sqr(J_p)*T_p);

        // Fix normal incidence
        auto normal_incidence_mask = Frame3f::cos_theta(wi) > Float(1.0 - 1e-4);
        dr::masked(T, normal_incidence_mask) = Float(0.5) * (T_s + T_p);

        return T;
    }


    template <int N>
    Vector3f compute_local_refraction_position_binary_search(Vector3f below_position, Vector3f above_position) const
    {
        // NOTE: assume that eta > 1!
        Ray3f ray(above_position, below_position - above_position);
        // Ray parameter from above point to below point where z = 0
        Float t_min = - ray.o.z() / ray.d.z();
        Float t_max = 1;
        // Extremes are straight line (t_min) and most extreme refraction possible (t_max).

        for (int i = 0; i < N; ++i)
        {
            Float t = Float(0.5)*(t_min + t_max);

            Vector3f intersection_position = ray(t);
            intersection_position.z() = 0;

            // NOTE: in Mitsuba the directions point away from the interactions!!
            Vector3f incoming_dir = dr::normalize(above_position - intersection_position);
            Vector3f outgoing_dir = dr::normalize(below_position - intersection_position);

            Float cos_theta_i = Frame3f::cos_theta(incoming_dir);
            auto [ F, cos_theta_t, eta_it, eta_ti ] = fresnel(cos_theta_i, m_eta);

            Vector3f refracted_dir = refract(incoming_dir, cos_theta_t, eta_ti);

            // We optimize towards zero, not -\infty!
            Float loss = outgoing_dir.z() - refracted_dir.z();

            t_max = dr::select(loss < 0, t, t_max);
            t_min = dr::select(loss < 0, t_min, t);
        }

        Float t = Float(0.5)*(t_min + t_max);
        Vector3f refraction_position = ray(t);
        refraction_position.z() = 0;
        return refraction_position;
    }



    void normalize_with_grad(Vector3f x, Vector3f dx, Vector3f &result, Vector3f &dresult) const
    {
        Float xdotdx = dr::dot(x, dx);
        Float xdotx = dr::dot(x, x);
        Float x_len = dr::norm(x);
        Vector3f x_norm = dr::normalize(x);

        result = x_norm;
        dresult = (dx * x_len - x_norm * xdotdx) / xdotx;
    }

    void local_refract_with_grad(Vector3f local_I, Float eta, Vector3f dlocal_I, Vector3f &local_result, Vector3f &dlocal_result) const
    {
        // TODO verify consistency with mitsuba refract!!
        // NOTE: this is _our_ refraction. (Inconsistent with mitsuba's refract)
        Float I_z = local_I.z();
        Float k = 1 - eta*eta * (1 - I_z*I_z);

        local_result = eta * local_I - (eta * I_z + dr::sqrt(k)) * Vector3f(0, 0, 1);
        dlocal_result = Vector3f(eta, eta, -eta*eta*I_z / dr::sqrt(k)) * dlocal_I; // diag(...) * dlocal_I

        local_result = dr::select(k > 0, local_result, dr::zeros<Vector3f>(0));
        dlocal_result = dr::select(k > 0, dlocal_result, dr::zeros<Vector3f>(0));
    }

    Float compute_local_refraction_position_jacobian(Vector3f local_frame_bottom_position, Vector3f local_frame_refraction_position, Vector3f local_frame_vector_to_shading_point) const
    {
        // Compute d(local_frame_bottom_position.xy) / d(local_frame_refraction_position.xy)
        // NOTE these are two variables!
        // TODO can we restrict to the 1d manifold??? probably not?

        Vector3f unnormalized_incoming_ray_dir = local_frame_refraction_position - local_frame_vector_to_shading_point;
        Vector3f dunnormalized_incoming_ray_dir_dx = Vector3f(1, 0, 0);
        Vector3f dunnormalized_incoming_ray_dir_dy = Vector3f(0, 1, 0);

        Vector3f incoming_ray_dir;
        Vector3f dincoming_ray_dir_dx;
        Vector3f dincoming_ray_dir_dy;
        normalize_with_grad(unnormalized_incoming_ray_dir, dunnormalized_incoming_ray_dir_dx, incoming_ray_dir, dincoming_ray_dir_dx);
        normalize_with_grad(unnormalized_incoming_ray_dir, dunnormalized_incoming_ray_dir_dy, incoming_ray_dir, dincoming_ray_dir_dy);

        Float eta_ti = dr::rcp(m_eta); // m_eta = n2/n1, eta_ti = n2/n1

        Vector3f below_ray_dir;
        Vector3f dbelow_ray_dir_dx;
        Vector3f dbelow_ray_dir_dy;
        local_refract_with_grad(incoming_ray_dir, eta_ti, dincoming_ray_dir_dx, below_ray_dir, dbelow_ray_dir_dx);
        local_refract_with_grad(incoming_ray_dir, eta_ti, dincoming_ray_dir_dy, below_ray_dir, dbelow_ray_dir_dy);

        // glm::vec3 local_frame_bottom_position = local_frame_refraction_position + below_ray_dir * local_frame_bottom_position.z / below_ray_dir.z
        Vector2f dlocal_frame_bottom_position_dx = Vector2f(1, 0) + (dr::head<2>(dbelow_ray_dir_dx) * below_ray_dir.z() - dr::head<2>(below_ray_dir) * dbelow_ray_dir_dx.z()) * local_frame_bottom_position.z() / (below_ray_dir.z() * below_ray_dir.z());
        Vector2f dlocal_frame_bottom_position_dy = Vector2f(0, 1) + (dr::head<2>(dbelow_ray_dir_dy) * below_ray_dir.z() - dr::head<2>(below_ray_dir) * dbelow_ray_dir_dy.z()) * local_frame_bottom_position.z() / (below_ray_dir.z() * below_ray_dir.z());

        Float dbottom_drefraction = dr::abs(dr::det(Matrix2f(dlocal_frame_bottom_position_dx, dlocal_frame_bottom_position_dy)));
        return dbottom_drefraction;
    }



    Spectrum eval(const SurfaceInteraction3f &si, Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::EndpointEvaluate, active);

        // Evaluate the Fresnel equations for unpolarized illumination
        Float cos_theta_i = Frame3f::cos_theta(si.wi);
        active &= cos_theta_i > 0;

        auto [ a_s, a_p, cos_theta_t, eta_it, eta_ti ] = fresnel_polarized(cos_theta_i, m_eta);
        auto cos_theta_t_abs = dr::abs(cos_theta_t);

        // Transmittance
        Spectrum T = compute_display_glass_transmittance(si.wi);

        // Emission profile
        Spectrum E = compute_display_emission_profile(cos_theta_t_abs);

        // NOTE: it is possible that the value returned by m_radiance->eval does _not_ depend on the uv coordinates but on the mesh attributes instead.
        // This will not be supported properly here.

        Vector3f wo = refract(si.wi, cos_theta_t, eta_ti);
        // Project wo to the "bottom" of the display
        //     t * wo.z() = -m_thickness
        // <=> t = -m_thickness / wo.z()
        //  => wo *= -m_thickness / wo.z()
        Vector2f projected_bottom_dir = -m_thickness * dr::head<2>(wo) / wo.z();

        // Also in local coordinate system
        Vector3f dp_du = si.to_local(si.dp_du);
        Vector3f dp_dv = si.to_local(si.dp_dv);

        // Fuv(Fp(uv)) = uv
        // D( Fuv(Fp(uv)) ) = Id = Duv(p) * Dp(uv)
        // Initialize from list of columns, just like in GLSL
        Matrix2f dp_duv = Matrix2f(dr::head<2>(dp_du), dr::head<2>(dp_dv));
        Matrix2f duv_dp = dr::inverse(dp_duv);

        Vector2f uv_shift = duv_dp * projected_bottom_dir;

        SurfaceInteraction3f si_bottom = si;
        si_bottom.uv += uv_shift;

        auto radiance = m_radiance->eval(si_bottom, active);
        return dr::select(active, T * E * radiance, 0);
    }

    std::pair<Ray3f, Spectrum> sample_ray(Float /*time*/, Float /*wavelength_sample*/,
                                          const Point2f &/*sample2*/, const Point2f &/*sample3*/,
                                          Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::EndpointSampleRay, active);

        // TODO sample direction on emitter surface and transform into scene
        return {};
    }

    std::pair<DirectionSample3f, Spectrum>
    sample_direction(const Interaction3f &it, const Point2f &sample, Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::EndpointSampleDirection, active);
        Assert(dynamic_cast<Rectangle*>(m_shape), "Can't sample from a display emitter without an associated Rectangle as shape!");
        DirectionSample3f ds;
        SurfaceInteraction3f si;

        // Importance sample the texture, then map onto the shape
        auto [ uv, uv_pdf ] = m_radiance->sample_position(sample, active);
        active &= dr::neq(uv_pdf, 0.f);

        si = m_shape->eval_parameterization(uv, +RayFlags::All, active);
        si.initialize_sh_frame(); // < without initializing the sh frame, the sh frame will be the to world transform of the shape! (including scaling factors!!)
        si.wavelengths = it.wavelengths;
        active &= si.is_valid();

        // In shading frame:
        Vector3f vector_to_shading_point = si.to_local(it.p - si.p);
        // If the shading point is on or below the horizon, let sampling fail!
        active &= !( vector_to_shading_point.z() < Float(1e-6) );

        Vector3f vector_to_bottom = Vector3f(0, 0, -m_thickness);

        // NOTE: this must happen in orthonormal coordinate system (local_frame), the local coordinate system of the rectangle might be scaled!
        ///glm::vec3 vector_to_refraction_ts = compute_local_refraction_position_newton<10>(local_frame_vector_to_bottom, local_frame_vector_to_shading_point, sbt_data->eta);
        Vector3f vector_to_refraction  = compute_local_refraction_position_binary_search<10>(vector_to_bottom, vector_to_shading_point);

        // Shift surface position in world space.
        si.p += si.to_world(vector_to_refraction);

        // TODO check if refraction position is out of local geometry bounds!!

        // Compute direction sample
        ds.p = si.p;
        ds.n = si.n;
        ds.uv = si.uv;
        ds.time = it.time;
        ds.delta = false;
        ds.d = ds.p - it.p; // TODO verify: direction pointing away from it

        Float dist_squared = dr::squared_norm(ds.d);
        ds.dist = dr::sqrt(dist_squared);
        ds.d /= ds.dist;

        // Multiply Jacobian determinant due to refraction shift
        Float dbottom_dsurface = compute_local_refraction_position_jacobian(vector_to_bottom, vector_to_refraction, vector_to_shading_point);

        // Compute final sampling pdf
        Float dp = dr::dot(ds.d, ds.n);
        active &= dp < -1e-3; //0.f;
        ds.pdf = dr::select(active, uv_pdf * dbottom_dsurface / dr::norm(dr::cross(si.dp_du, si.dp_dv)) *
                                    dist_squared / -dp, 0.f);

        // Compute radiance / pdf

        Vector3f wi = si.to_local(-ds.d); // ds.d points towards si, wi points away from si!

        // Evaluate the Fresnel equations for unpolarized illumination
        Float cos_theta_i = Frame3f::cos_theta(wi);
        auto [ a_s, a_p, cos_theta_t, eta_it, eta_ti ] = fresnel_polarized(cos_theta_i, m_eta);
        auto cos_theta_t_abs = dr::abs(cos_theta_t);

        // Transmittance
        Spectrum T = compute_display_glass_transmittance(wi);

        // Emission profile
        Spectrum E = compute_display_emission_profile(cos_theta_t_abs);

        Spectrum radiance = T * E * m_radiance->eval(si, active) / ds.pdf;

        ds.emitter = this;
        return { ds, radiance & active };
    }

    Float pdf_direction(const Interaction3f &it, const DirectionSample3f &ds,
                        Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::EndpointEvaluate, active);

        // Get the surface interaction!
        SurfaceInteraction3f si = m_shape->eval_parameterization(ds.uv, +RayFlags::All, active);
        si.initialize_sh_frame(); // < without initializing the sh frame, the sh frame will be the to world transform of the shape! (including scaling factors!!)
        // Note: Shading frame will be identical for all positions on the shape!
        Vector3f vector_to_shading_point = si.to_local(it.p - si.p);

        // Evaluate the Fresnel equations for unpolarized illumination
        Vector3f wi = dr::normalize(-vector_to_shading_point);
        Float cos_theta_i = Frame3f::cos_theta(wi);
        active &= cos_theta_i > 0;
        auto [ a_s, a_p, cos_theta_t, eta_it, eta_ti ] = fresnel_polarized(cos_theta_i, m_eta);
        Vector3f wo = refract(wi, cos_theta_t, eta_ti);

        // Project refracted direction to the "bottom" of the display
        Vector3f vector_to_bottom = (-m_thickness / wo.z()) * wo;
        Vector2f projected_bottom_dir = -m_thickness * dr::head<2>(wo) / wo.z();

        // Also in local coordinate system
        Vector3f dp_du = si.to_local(si.dp_du);
        Vector3f dp_dv = si.to_local(si.dp_dv);

        // Fuv(Fp(uv)) = uv
        // D( Fuv(Fp(uv)) ) = Id = Duv(p) * Dp(uv)
        // Initialize from list of columns, just like in GLSL
        Matrix2f dp_duv = Matrix2f(dr::head<2>(dp_du), dr::head<2>(dp_dv));
        Matrix2f duv_dp = dr::inverse(dp_duv);

        Vector2f uv_shift = duv_dp * projected_bottom_dir;
        Point2f uv = si.uv + uv_shift;
        Float uv_pdf = m_radiance->pdf_position(uv, active);

        // multiply this to position sampling pdf:
        Vector3f refraction_position = dr::zeros<Vector3f>();
        Float dbottom_dsurface = compute_local_refraction_position_jacobian(vector_to_bottom, refraction_position, vector_to_shading_point);

        // Compute final sampling pdf
        Float dp = dr::dot(ds.d, ds.n);
        active &= dp < -1e-3;//0.f;
        Float pdf = dr::select(active, uv_pdf * dbottom_dsurface / dr::norm(dr::cross(si.dp_du, si.dp_dv)) *
                                    (ds.dist * ds.dist) / -dp, 0.f);

        return pdf;
    }

    ScalarBoundingBox3f bbox() const override { return m_shape->bbox(); }

    void traverse(TraversalCallback *callback) override {
        // We don't want to support differentiation wrt. display parameters right now...
        callback->put_object("radiance", m_radiance.get(), +ParamFlags::NonDifferentiable);
        callback->put_parameter("thickness", m_thickness, +ParamFlags::NonDifferentiable);
        callback->put_parameter("eta", m_eta, +ParamFlags::NonDifferentiable);
        callback->put_parameter("polarization_dir", m_polarization_dir, +ParamFlags::NonDifferentiable);
        callback->put_parameter("emission_coeffs", m_emission_coeffs, +ParamFlags::NonDifferentiable);
    }

    std::string to_string() const override {
        std::ostringstream oss;
        oss << "DisplayEmitter[" << std::endl;
        oss << std::endl << "]";
        return oss.str();
    }

    MI_DECLARE_CLASS()

private:
    Float  m_thickness;
    Float  m_eta;
    Vector2f  m_polarization_dir;

    std::array<Float, 6>  m_emission_coeffs;

    ref<Texture> m_radiance;

};

MI_IMPLEMENT_CLASS_VARIANT(DisplayEmitter, Emitter)
MI_EXPORT_PLUGIN(DisplayEmitter, "Display emitter")
NAMESPACE_END(mitsuba)
