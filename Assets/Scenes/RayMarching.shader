// Upgrade NOTE: replaced 'mul(UNITY_MATRIX_MVP,*)' with 'UnityObjectToClipPos(*)'

Shader "Hidden/RayMarching"
{
    Properties
    {
        _MainTex ("Texture", 2D) = "white" {}
    }
    SubShader
    {
        // No culling or depth
        Cull Off ZWrite Off ZTest Always

        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag

            #include "UnityCG.cginc"

            // Provided by our script
            uniform float4x4 _FrustumCornersES;
            uniform sampler2D _MainTex;
            uniform float4 _MainTex_TexelSize;
            uniform float4x4 _CameraInvViewMatrix;
            uniform float3 _CameraWS;
            uniform float3 _LightDir;
            uniform sampler2D _CameraDepthTexture;
            uniform sampler2D _ColorRamp;
            uniform sampler2D _ColorSky;

            struct appdata
            {
                float4 vertex : POSITION;
                float2 uv : TEXCOORD0;
            };

            // struct v2f
            // {
            //     float2 uv : TEXCOORD0;
            //     float4 vertex : SV_POSITION;
            // };
            // Output of vertex shader / input to fragment shader
            struct v2f
            {
                float4 pos : SV_POSITION;
                float2 uv : TEXCOORD0;
                float3 ray : TEXCOORD1;
            };

            v2f vert (appdata v)
            {
                v2f o;
                // o.vertex = UnityObjectToClipPos(v.vertex);
                // o.uv = v.uv;

                 // Index passed via custom blit function in RaymarchGeneric.cs
                half index = v.vertex.z;
                v.vertex.z = 0.1;
                
                o.pos = UnityObjectToClipPos(v.vertex);
                o.uv = v.uv.xy;
                
                #if UNITY_UV_STARTS_AT_TOP
                if (_MainTex_TexelSize.y < 0)
                    o.uv.y = 1 - o.uv.y;
                #endif

                // Get the eyespace view ray (normalized)
                o.ray = _FrustumCornersES[(int)index].xyz;

                // Dividing by z "normalizes" it in the z axis
                // Therefore multiplying the ray by some number i gives the viewspace position
                // of the point on the ray with [viewspace z]=i
                o.ray /= abs(o.ray.z);

                // Transform the ray from eyespace to worldspace
                // Note: _CameraInvViewMatrix was provided by the script
                o.ray = mul(_CameraInvViewMatrix, o.ray);

                // Start magic with space spin
                // float3 diff = float3(0,0,0) -  (o.pos.xyz + o.ray );

                // o.ray += 0.5 * diff*(1 / (length(diff) + 0.0001));

                return o;
            }

            // Torus
            // t.x: diameter
            // t.y: thickness
            // Adapted from: http://iquilezles.org/www/articles/distfunctions/distfunctions.htm
            float sdTorus(float3 p, float2 t)
            {
                float2 q = float2(length(p.xz) - t.x, p.y);
                return length(q) - t.y;
            }

            float sphere(float3 p, float4 s){
                return length(p - s.xyz) - s.w;
            }

            // Box
            // b: size of box in x/y/z
            // Adapted from: http://iquilezles.org/www/articles/distfunctions/distfunctions.htm
            float sdBox(float3 p, float3 b)
            {
                float3 d = abs(p) - b;
                return min(max(d.x, max(d.y, d.z)), 0.0) +
                    length(max(d, 0.0));
            }

            float sdRoundedCylinder( float3 p, float ra, float rb, float h )
            {
                float2 d = float2( length(p.xz)-2.0*ra+rb, abs(p.y) - h );
                return min(max(d.x,d.y),0.0) + length(max(d,0.0)) - rb;
            }

            // Union
            // Adapted from: http://iquilezles.org/www/articles/distfunctions/distfunctions.htm
            float2 opU( float2 d1, float2 d2 )
            {   
                return (d1.x < d2.x) ? d1 : d2;
            }

            // Subtraction
            // Adapted from: http://iquilezles.org/www/articles/distfunctions/distfunctions.htm
            float opS( float d1, float d2 )
            {
                return max(-d1,d2);
            }

            // Intersection
            // Adapted from: http://iquilezles.org/www/articles/distfunctions/distfunctions.htm
            float opI( float d1, float d2 )
            {
                return max(d1,d2);
            }

            float2 map(float3 p) {
                // float union_box = opU(
                //     sdBox(p - float3(-4.5, 0.5, 0), float3(1,1,1)), 
                //     sdBox(p - float3(-3.5, -0.5, 0), float3(1,1,1))
                // );
                // float subtr_box = opS(
                //     sdBox(p - float3(-0.5, 0.5, 0), float3(1,1,1.01)), 
                //     sdBox(p - float3(0.5, -0.5, 0), float3(1,1,1))
                // );
                // float insec_box = opI(
                //     sdBox(p - float3(3.5, 0.5, 0), float3(1,1,1)), 
                //     sdBox(p - float3(4.5, -0.5, 0), float3(1,1,1))
                // );

                // float ret = opU(union_box, subtr_box);
                // ret = opU(ret, insec_box);

                // float ret = opU(sdTorus(p, float2(2, 0.2)), sdTorus(p, float2(1, 0.2)));
                float2 sphere1 = float2(sphere(p, float4(0, 0, 0, 0.7)), 0.99);
                float2 cylinder1 = float2(sdRoundedCylinder(p , 0.7, 0.01, 0.001), 0.005);
                float2 ret = opU(sphere1, cylinder1);
                return ret;
            }

            float2 cylinder1(float3 p) {
                return float2(sdRoundedCylinder(p , 0.7, 0.01, 0.001), 0.005);
            }

            float3 calcNormal(in float3 pos)
            {
                // epsilon - used to approximate dx when taking the derivative
                const float2 eps = float2(0.001, 0.0);

                // The idea here is to find the "gradient" of the distance field at pos
                // Remember, the distance field is not boolean - even if you are inside an object
                // the number is negative, so this calculation still works.
                // Essentially you are approximating the derivative of the distance field at this point.
                float3 nor = float3(
                    map(pos + eps.xyy).x - map(pos - eps.xyy).x,
                    map(pos + eps.yxy).x - map(pos - eps.yxy).x,
                    map(pos + eps.yyx).x - map(pos - eps.yyx).x);
                return normalize(nor);
            }

            // Raymarch along given ray
            // ro: ray origin
            // rd: ray direction
            fixed4 raymarch(float3 ro, float3 rd, float s) {
                fixed4 ret = fixed4(0,0,0,0);

                const int maxstep = 64;
                const float drawdist = 40; // draw distance in unity units

                float t = 0.1; // current distance traveled along ray

                
                for (int i = 0; i < maxstep; ++i) {

                    // If we run past the depth buffer, stop and return nothing (transparent pixel)
                    // this way raymarched objects and traditional meshes can coexist.
                    if (t >= s || t > drawdist) {
                        ret = fixed4(0,0,0,0);
                        break;
                    }
                    // float3 p = ro + rd * t; // World space position of sample
                    // Start magic with black hole
                    float3 diff = float3(0,0,0)- (ro + rd*t); // diff to gravity
                    float dist = length(diff);
                    diff = normalize(diff);
                    diff = 0.007/(dist*dist) * diff;

                    // Учитываем искривление луча
                    rd = normalize(rd+diff);

                    float3 p = ro + rd*t;
                    float2 d = map(p);       // Sample of distance field (see map())

                    // Устаналиваем новую исходную точку
                    ro = p;
                    rd = normalize(rd+diff);

                    // If the sample <= 0, we have hit something (see map()).
                    if (d.x < 0.001) {
                        // Lambertian Lighting
                        float3 n = calcNormal(p);
                        
                        float light = dot(-_LightDir.xyz, n);
                        if (d.y < 0.5){
                            d.y = d.y + 0.1*length(p)*0.02;
                        }
                        ret = fixed4(tex2D(_ColorRamp, float2(d.y, 0.7)).xyz*light, 1);
                        break;
                    }

                    // If the sample > 0, we haven't hit anything yet so we should march forward
                    // We step forward by distance d, because d is the minimum distance possible to intersect
                    // an object (see map()).
                    // t += d;
                    t = d.x;
                }
                return ret;
            }

            fixed4 frag (v2f i) : SV_Target
            {
                // fixed4 col = tex2D(_MainTex, i.uv);
                // // just invert the colors
                // col.rgb = 1 - col.rgb;

                // fixed4 col = fixed4(i.ray, 1);
                // return col;

                // ray direction
                float3 rd = normalize(i.ray.xyz);
                // ray origin (camera position)
                float3 ro = _CameraWS;

                float2 duv = i.uv;
                #if UNITY_UV_STARTS_AT_TOP
                if (_MainTex_TexelSize.y < 0)
                    duv.y = 1 - duv.y;
                #endif
                
                // Convert from depth buffer (eye space) to true distance from camera
                // This is done by multiplying the eyespace depth by the length of the "z-normalized"
                // ray (see vert()).  Think of similar triangles: the view-space z-distance between a point
                // and the camera is proportional to the absolute distance.
                float depth = LinearEyeDepth(tex2D(_CameraDepthTexture, duv).r);
                depth *= length(i.ray.xyz);
                // change u,v

                fixed4 add = raymarch(ro, rd, depth);

                // find ray changing
                const int maxstep = 200;
                float t_ = 0.1; // current distance traveled along ray 
                float min_dist = 1000;
                float min_dist1;
                float min_dist2;
                
                for (int i = 0; i < maxstep; ++i) {


                    // Start magic with black hole
                    float3 diff = float3(0,0,0)- (ro + rd*t_); // diff to gravity
                    float dist = length(diff);
                    diff = normalize(diff);
                    diff = 0.007/(dist*dist) * diff;

                    // Учитываем искривление луча
                    rd = normalize(rd+diff);
                    float3 p = ro + (rd)*t_;
                    float d = cylinder1(p);
                    if (abs(d) < min_dist){
                        min_dist = d;
                    }

                    // Устаналиваем новую исходную точку
                    ro = p;
                    //rd = normalize(rd+diff);
                }

                // чтобы нуле не равнялось расстояние:
                min_dist1 = min_dist;
                if( min_dist < 0.1) {
                    min_dist1 = 0.1;
                }

                // ro = ro + rd*t_*maxstep + totaldiff * t_*maxstep;
                // rd = normalize(rd+totaldiff);





                // let us try to fing sphere intersection
                float dot_prod = dot(ro, rd);
                float r = 1000;
                float d = length(ro);
                float t = - dot_prod + sqrt(dot_prod * dot_prod + r*r -d*d);
                float3 intersect = ro + rd*t;
                intersect = normalize(intersect);
                // get angles of intersection
                float teta = asin(intersect.y) / (3.14) + 1./2;
                float dzeta = acos(intersect.x);
                if (intersect.z < 0) {
                    dzeta = 2 * 3.14 - dzeta; 
                }
                dzeta = dzeta / (2*3.14);

                float2 new_uv = float2(teta, dzeta);
                //fixed3 col = tex2D(_MainTex,i.uv); // Color of the scene before this shader was run
                fixed3 col = tex2D(_ColorSky, new_uv); // Color of the scene before this shader was run

                col = col + col*(0.7/min_dist1);

                //fixed3 col = tex2D(_MainTex,new_uv); // Color of the scene before this shader was run
                //col = col + float3(1,1,1)*(0.0000001/min_dist);
                
                // чтобы нуле не равнялось расстояние:
                // min_dist2 = min_dist;
                // if( min_dist < 0.02) {
                //     min_dist2 = 0.02;
                // }
                // add.xyz = float3(0.5,0.5,0.5) *(0.002/min_dist2);
                
                // Returns final color using alpha blending
                col = col*(1.0 - add.w) + add.xyz * add.w;

                // if (col.x > 1) {col.x = 1;}
                // if (col.y > 1) {col.y = 1;}
                // if (col.z > 1) {col.z = 1;}
                return fixed4(col, 1.0);
            }
            ENDCG
        }
    }
}
