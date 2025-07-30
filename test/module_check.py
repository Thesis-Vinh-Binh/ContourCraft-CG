# enable_addon_debug.py
import bpy

# Try enabling the addon
try:
    bpy.ops.preferences.addon_enable(module='smplx_blender_addon')
    print("[INFO] smplx_blender_addon enabled.")
except Exception as e:
    print(f"[ERROR] Could not enable addon: {e}")

# List all enabled addons
print("Enabled Add-ons:")
for addon in bpy.context.preferences.addons.keys():
    print(" -", addon)

# Optional: List all object operators
print("\nAvailable bpy.ops.object operators:")
for op in dir(bpy.ops.object):
    if op.startswith("smplx") or "smplx" in op:
        print(" -", op)
