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

                // Transform the ray from eyespace to worldspace
                // Note: _CameraInvViewMatrix was provided by the script
                o.ray = mul(_CameraInvViewMatrix, o.ray);

                return o;
            }

            fixed4 frag (v2f i) : SV_Target
            {
                // fixed4 col = tex2D(_MainTex, i.uv);
                // // just invert the colors
                // col.rgb = 1 - col.rgb;

                fixed4 col = fixed4(i.ray, 1);
                return col;


            }
            ENDCG
        }
    }
}
