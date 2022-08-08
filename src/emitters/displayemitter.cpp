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
        Float cos_theta_i_abs = dr::abs(cos_theta_i);

        auto [ a_s, a_p, cos_theta_t, eta_it, eta_ti ] = fresnel_polarized(cos_theta_i, m_eta);
        auto cos_theta_t_abs = dr::abs(cos_theta_t);

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


    Spectrum eval(const SurfaceInteraction3f &si, Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::EndpointEvaluate, active);

        // Evaluate the Fresnel equations for unpolarized illumination
        Float cos_theta_i = Frame3f::cos_theta(si.wi);
        Float cos_theta_i_abs = dr::abs(cos_theta_i);

        /* Using Snell's law, calculate the squared sine of the
        angle between the surface normal and the transmitted ray */
        //Float cos_theta_t_sqr = 1.0f - m_eta * m_eta * (1.0f - cos_theta_i * cos_theta_i);
        //Float cos_theta_t = mulsign_neg(cos_theta_t_abs, cos_theta_i);

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
        return T * E * radiance;
    }

    std::pair<Ray3f, Spectrum> sample_ray(Float time, Float wavelength_sample,
                                          const Point2f &sample2, const Point2f &sample3,
                                          Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::EndpointSampleRay, active);

        // TODO sample direction on emitter surface and transform into scene
        return {};
    }

    std::pair<DirectionSample3f, Spectrum>
    sample_direction(const Interaction3f &it, const Point2f &sample, Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::EndpointSampleDirection, active);

        // TODO NEE manifold exploration here...
        return {};
    }

    Float pdf_direction(const Interaction3f &it, const DirectionSample3f &ds,
                        Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::EndpointEvaluate, active);

        // TODO evaluate pdf for sampled direction
        // this might be difficult!!
        return 0; // TODO
    }

    ScalarBoundingBox3f bbox() const override { return m_shape->bbox(); }

    void traverse(TraversalCallback *callback) override {
        // We don't want to support differentiation wrt. display parameters right now...
        callback->put_object("radiance", m_radiance.get(), +ParamFlags::NonDifferentiable);
        callback->put_parameter("thickness", m_thickness, +ParamFlags::NonDifferentiable);
        callback->put_parameter("eta", m_eta, +ParamFlags::NonDifferentiable);
        callback->put_parameter("polarization_dir", m_polarization_dir, +ParamFlags::NonDifferentiable);
        callback->put_parameter("emission_coeffs", m_emission_coeffs, +ParamFlags::NonDifferentiable);
        /*
        callback->put_parameter("emission_coeffs[0]", m_emission_coeffs[0]);
        callback->put_parameter("emission_coeffs[1]", m_emission_coeffs[1]);
        callback->put_parameter("emission_coeffs[2]", m_emission_coeffs[2]);
        callback->put_parameter("emission_coeffs[3]", m_emission_coeffs[3]);
        callback->put_parameter("emission_coeffs[4]", m_emission_coeffs[4]);
        callback->put_parameter("emission_coeffs[5]", m_emission_coeffs[5]);
        */
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
