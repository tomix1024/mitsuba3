#include <mitsuba/render/sensor.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/core/transform.h>
#include <mitsuba/core/bbox.h>

NAMESPACE_BEGIN(mitsuba)

/**!

.. _sensor-opencv:

OpenCV pinhole camera (:monosp:`opencv`)
--------------------------------------------------

.. pluginparameters::
 :extra-rows: 7

 * - to_world
   - |transform|
   - Specifies an optional camera-to-world transformation.
     (Default: none (i.e. camera space = world space))
   - |exposed|

 * - fx, fy
   - |float|
   - Denotes the :monosp:`focal_length` values of the camera matrix.

 * - cx, cy
   - |float|
   - Denotes :monosp:`principal point` of the camera matrix.

 * - near_clip, far_clip
   - |float|
   - Distance to the near/far clip planes. (Default: :monosp:`near_clip=1e-2` (i.e. :monosp:`0.01`)
     and :monosp:`far_clip=1e4` (i.e. :monosp:`10000`))
   - |exposed|

 * - srf
   - |spectrum|
   - Sensor Response Function that defines the :ref:`spectral sensitivity <explanation_srf_sensor>`
     of the sensor (Default: :monosp:`none`)

 */

template <typename Float, typename Spectrum>
class OpenCVCamera final : public ProjectiveCamera<Float, Spectrum> {
public:
    MI_IMPORT_BASE(ProjectiveCamera, m_to_world, m_needs_sample_3,
                    m_film, m_sampler, m_resolution, m_shutter_open,
                    m_shutter_open_time, m_near_clip, m_far_clip,
                    sample_wavelengths)
    MI_IMPORT_TYPES()

    OpenCVCamera(const Properties &props) : Base(props) {
        ScalarVector2i size = m_film->size();

        m_focal_lengths = ScalarPoint2f(
            props.get<ScalarFloat>("fx", 1.f),
            props.get<ScalarFloat>("fy", 1.f)
        );

        m_principal_point = ScalarPoint2f(
            props.get<ScalarFloat>("cx", 0.f),
            props.get<ScalarFloat>("cy", 0.f)
        );

        if (m_to_world.scalar().has_scale())
            Throw("Scale factors in the camera-to-world transformation are not allowed!");

        update_camera_transforms();
    }

    void traverse(TraversalCallback *callback) override {
        Base::traverse(callback);
        callback->put_parameter("focal_lengths", m_focal_lengths, +ParamFlags::NonDifferentiable);
        callback->put_parameter("principal_point", m_principal_point, +ParamFlags::NonDifferentiable);
        callback->put_parameter("to_world", *m_to_world.ptr(), +ParamFlags::NonDifferentiable);
    }

    void parameters_changed(const std::vector<std::string> &keys) override {
        if (keys.empty() || string::contains(keys, "to_world")) {
            // Update the scalar value of the matrix
            m_to_world = m_to_world.value();
            if (m_to_world.scalar().has_scale())
                Throw("Scale factors in the camera-to-world transformation are not allowed!");
        }

        Base::parameters_changed(keys);
        update_camera_transforms();
    }

    Transform4f compute_camera_to_sample() {
        /*
        return perspective_projection(
            m_film->size(), m_film->crop_size(), m_film->crop_offset(),
            m_x_fov, m_near_clip, m_far_clip);
        */

        Vector2f film_size = Vector2f(Vector<int, 2>(m_film->size()));

        Float aspect = film_size.x() / film_size.y();

        Vector2f crop_size = Vector2f(Vector<int, 2>(m_film->crop_size()));
        Vector2f crop_offset = Vector2f(Vector<int, 2>(m_film->crop_offset()));

        Vector2f rel_size    = crop_size / film_size,
                 rel_offset  = crop_offset / film_size;

        Transform4f filmuv_to_cropuv_transform =
            Transform4f::scale(
               Vector3f(1.f / rel_size.x(), 1.f / rel_size.y(), 1.f)) *
            Transform4f::translate(
               Vector3f(-rel_offset.x(), -rel_offset.y(), 0.f));

        /*
        Transform4f scale_image = Transform4f::scale(Vector3f(-0.5f, -0.5f * aspect, 1.f));
        Transform4f translate_image = Transform4f::translate(Vector3f(-1.f, -1.f / aspect, 0.f));

        Transform4f perspective = Transform4f::perspective(m_x_fov, m_near_clip, m_far_clip);
        */

        Transform4f image_to_filmuv_transform =
            Transform4f::scale(
                Vector3f(1.0f / film_size.x(), 1.0f / film_size.y(), 1.0f)) *
            // We need a small translation since Mitsuba starts the pixel indices in the corner, and opencv starts the pixel indices at the pixel centers!
            // Shift by half a pixel! (image corner (-0.5,-0.5) maps to (0, 0)
            Transform4f::translate(
                Vector3f(0.5f, 0.5f, 0.0f));
        /*
            Transform4f::translate(
                Vector3f(-1.0f, -1.0f, 0.0f)) *
            Transform4f::scale(
                Vector3f(2.0f / film_size.x(), 2.0f / film_size.y(), 1.0f));
        */


        Float recip = 1.f / (m_far_clip - m_near_clip);
        Matrix4f opencv_camera_mat = Matrix4f(
            // Layout matches matrix layout:
            m_focal_lengths.x(), 0.f, m_principal_point.x(), 0.f,
            0.f, m_focal_lengths.y(), m_principal_point.y(), 0.f,
            0.f, 0.f, m_far_clip * recip, -m_near_clip * m_far_clip * recip,
            0.f, 0.f, 1.f, 0.f
        );
        Transform4f opencv_camera_to_image_transform = Transform4f(opencv_camera_mat);

        //Transform4f axis_flip_transform = Transform4f::scale(Vector3f(-1.0f, -1.0f, 1.f));

        return
           filmuv_to_cropuv_transform *
           image_to_filmuv_transform *
           opencv_camera_to_image_transform; // * axis_flip_transform,
    }

    void update_camera_transforms() {
        m_camera_to_sample = compute_camera_to_sample();

        m_sample_to_camera = m_camera_to_sample.inverse();

        // Position differentials on the near plane
        m_dx = m_sample_to_camera * Point3f(1.f / m_resolution.x(), 0.f, 0.f) -
               m_sample_to_camera * Point3f(0.f);
        m_dy = m_sample_to_camera * Point3f(0.f, 1.f / m_resolution.y(), 0.f)
             - m_sample_to_camera * Point3f(0.f);

        /* Precompute some data for importance(). Please
           look at that function for further details. */
        Point3f pmin(m_sample_to_camera * Point3f(0.f, 0.f, 0.f)),
                pmax(m_sample_to_camera * Point3f(1.f, 1.f, 0.f));

        m_image_rect.reset();
        m_image_rect.expand(Point2f(pmin.x(), pmin.y()) / pmin.z());
        m_image_rect.expand(Point2f(pmax.x(), pmax.y()) / pmax.z());
        m_normalization = 1.f / m_image_rect.volume();
        m_needs_sample_3 = false;

        dr::make_opaque(m_to_world, m_camera_to_sample, m_sample_to_camera, m_dx, m_dy, m_focal_lengths, m_principal_point,
                        m_image_rect, m_normalization);
    }

    std::pair<Ray3f, Spectrum> sample_ray(Float time, Float wavelength_sample,
                                          const Point2f &position_sample,
                                          const Point2f & /*aperture_sample*/,
                                          Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::EndpointSampleRay, active);

        auto [wavelengths, wav_weight] =
            sample_wavelengths(dr::zeros<SurfaceInteraction3f>(),
                               wavelength_sample,
                               active);
        Ray3f ray;
        ray.time = time;
        ray.wavelengths = wavelengths;

        // Compute the sample position on the near plane (local camera space).
        Point3f near_p = m_sample_to_camera *
                         Point3f(position_sample.x(), position_sample.y(), 0.f);

        // Convert into a normalized ray direction; adjust the ray interval accordingly.
        Vector3f d = dr::normalize(Vector3f(near_p));

        ray.o = m_to_world.value().translation();
        ray.d = m_to_world.value() * d;

        Float inv_z = dr::rcp(d.z());
        Float near_t = m_near_clip * inv_z,
              far_t  = m_far_clip * inv_z;
        ray.o += ray.d * near_t;
        ray.maxt = far_t - near_t;

        return { ray, wav_weight };
    }

    std::pair<RayDifferential3f, Spectrum>
    sample_ray_differential(Float time, Float wavelength_sample, const Point2f &position_sample,
                            const Point2f & /*aperture_sample*/, Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::EndpointSampleRay, active);

        auto [wavelengths, wav_weight] =
            sample_wavelengths(dr::zeros<SurfaceInteraction3f>(),
                               wavelength_sample,
                               active);
        RayDifferential3f ray;
        ray.time = time;
        ray.wavelengths = wavelengths;

        // Compute the sample position on the near plane (local camera space).
        Point3f near_p = m_sample_to_camera *
                         Point3f(position_sample.x(), position_sample.y(), 0.f);

        // Convert into a normalized ray direction; adjust the ray interval accordingly.
        Vector3f d = dr::normalize(Vector3f(near_p));

        ray.o = m_to_world.value().translation();
        ray.d = m_to_world.value() * d;

        Float inv_z = dr::rcp(d.z());
        Float near_t = m_near_clip * inv_z,
              far_t  = m_far_clip * inv_z;
        ray.o += ray.d * near_t;
        ray.maxt = far_t - near_t;

        ray.o_x = ray.o_y = ray.o;

        ray.d_x = m_to_world.value() * dr::normalize(Vector3f(near_p) + m_dx);
        ray.d_y = m_to_world.value() * dr::normalize(Vector3f(near_p) + m_dy);
        ray.has_differentials = true;

        return { ray, wav_weight };
    }

    std::pair<DirectionSample3f, Spectrum>
    sample_direction(const Interaction3f &it, const Point2f & /*sample*/,
                     Mask active) const override {
        // Transform the reference point into the local coordinate system
        Transform4f trafo = m_to_world.value();
        Point3f ref_p     = trafo.inverse().transform_affine(it.p);

        // Check if it is outside of the clip range
        DirectionSample3f ds = dr::zeros<DirectionSample3f>();
        ds.pdf = 0.f;
        active &= (ref_p.z() >= m_near_clip) && (ref_p.z() <= m_far_clip);
        if (dr::none_or<false>(active))
            return { ds, dr::zeros<Spectrum>() };

        Point3f screen_sample = m_camera_to_sample * ref_p;
        ds.uv = Point2f(screen_sample.x(), screen_sample.y());
        active &= (ds.uv.x() >= 0) && (ds.uv.x() <= 1) && (ds.uv.y() >= 0) &&
                  (ds.uv.y() <= 1);
        if (dr::none_or<false>(active))
            return { ds, dr::zeros<Spectrum>() };

        ds.uv *= m_resolution;

        Vector3f local_d(ref_p);
        Float dist     = dr::norm(local_d);
        Float inv_dist = dr::rcp(dist);
        local_d *= inv_dist;

        ds.p    = trafo.transform_affine(Point3f(0.0f));
        ds.d    = (ds.p - it.p) * inv_dist;
        ds.dist = dist;
        ds.n    = trafo * Vector3f(0.0f, 0.0f, 1.0f);
        ds.pdf  = dr::select(active, Float(1.f), Float(0.f));

        return { ds, Spectrum(importance(local_d) * inv_dist * inv_dist) };
    }

    ScalarBoundingBox3f bbox() const override {
        ScalarPoint3f p = m_to_world.scalar() * ScalarPoint3f(0.f);
        return ScalarBoundingBox3f(p, p);
    }

    /**
     * \brief Compute the directional sensor response function of the camera
     * multiplied with the cosine foreshortening factor associated with the
     * image plane
     *
     * \param d
     *     A normalized direction vector from the aperture position to the
     *     reference point in question (all in local camera space)
     */
    Float importance(const Vector3f &d) const {
        /* How is this derived? Imagine a hypothetical image plane at a
           distance of d=1 away from the pinhole in camera space.

           Then the visible rectangular portion of the plane has the area

              A = (2 * dr::tan(0.5 * xfov in radians))^2 / aspect

           Since we allow crop regions, the actual visible area is
           potentially reduced:

              A' = A * (cropX / filmX) * (cropY / filmY)

           Perspective transformations of such aligned rectangles produce
           an equivalent scaled (but otherwise undistorted) rectangle
           in screen space. This means that a strategy, which uniformly
           generates samples in screen space has an associated area
           density of 1/A' on this rectangle.

           To compute the solid angle density of a sampled point P on
           the rectangle, we can apply the usual measure conversion term:

              d_omega = 1/A' * distance(P, origin)^2 / dr::cos(theta)

           where theta is the angle that the unit direction vector from
           the origin to P makes with the rectangle. Since

              distance(P, origin)^2 = Px^2 + Py^2 + 1

           and

              dr::cos(theta) = 1/sqrt(Px^2 + Py^2 + 1),

           we have

              d_omega = 1 / (A' * cos^3(theta))
        */

        Float ct     = Frame3f::cos_theta(d),
              inv_ct = dr::rcp(ct);

        // Compute the position on the plane at distance 1
        Point2f p(d.x() * inv_ct, d.y() * inv_ct);

        /* Check if the point lies to the front and inside the
           chosen crop rectangle */
        Mask valid = ct > 0 && m_image_rect.contains(p);

        return dr::select(valid, m_normalization * inv_ct * inv_ct * inv_ct, 0.f);
    }

    std::string to_string() const override {
        using string::indent;

        std::ostringstream oss;
        oss << "OpenCVCamera[" << std::endl
            << "  focal_lengths = " << m_focal_lengths << "," << std::endl
            << "  principal_point = " << m_principal_point << "," << std::endl
            << "  near_clip = " << m_near_clip << "," << std::endl
            << "  far_clip = " << m_far_clip << "," << std::endl
            << "  film = " << indent(m_film) << "," << std::endl
            << "  sampler = " << indent(m_sampler) << "," << std::endl
            << "  resolution = " << m_resolution << "," << std::endl
            << "  shutter_open = " << m_shutter_open << "," << std::endl
            << "  shutter_open_time = " << m_shutter_open_time << "," << std::endl
            << "  to_world = " << indent(m_to_world, 13) << std::endl
            << "]";
        return oss.str();
    }

    MI_DECLARE_CLASS()
private:
    Transform4f m_camera_to_sample;
    Transform4f m_sample_to_camera;
    BoundingBox2f m_image_rect;
    Float m_normalization;
    Vector3f m_dx, m_dy;
    // Camera matrix parameters:
    Vector2f m_focal_lengths;
    Vector2f m_principal_point;
};

MI_IMPLEMENT_CLASS_VARIANT(OpenCVCamera, ProjectiveCamera)
MI_EXPORT_PLUGIN(OpenCVCamera, "OpenCV Camera");
NAMESPACE_END(mitsuba)
