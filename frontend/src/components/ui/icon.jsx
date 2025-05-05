"use client"

import * as React from "react"
import { cva } from "class-variance-authority"
import { cn } from "@/lib/utils"
import 'material-symbols';

const IconVariants = cva(
  "material-symbols-outlined transition-colors",
  {
    variants: {
      variant: {
        default: "text-foreground",
        primary: "text-primary",
        secondary: "text-secondary",
        destructive: "text-destructive",
        muted: "text-muted-foreground",
        accent: "text-accent-foreground",
        inherit: "", // Empty string to allow color inheritance from parent
      },
      size: {
        default: "text-base",
        xs: "text-xs",
        sm: "text-sm",
        md: "text-base",
        lg: "text-lg",
        xl: "text-xl",
        "2xl": "text-2xl",
      },
      style: {
        outlined: "material-symbols-outlined",
        rounded: "material-symbols-rounded",
        sharp: "material-symbols-sharp",
        filled: "material-symbols-filled",
      },
    },
    defaultVariants: {
      variant: "inherit", // Changed default to inherit parent color
      size: "default",
      style: "outlined",
    },
  }
)

function Icon({
  className,
  variant,
  size,
  style,
  weight,
  fill,
  grad,
  opsz,
  children,
  ...props
}) {
  // Create a string that combines all font-variation-settings
  const variationSettings = {};

  if (weight) variationSettings.wght = weight;
  if (fill !== undefined) variationSettings.FILL = fill;
  if (grad) variationSettings.GRAD = grad;
  if (opsz) variationSettings.opsz = opsz;

  // Convert variation settings object to proper CSS font-variation-settings string
  const fontVariationStyle = Object.keys(variationSettings).length
    ? {
        fontVariationSettings: Object.entries(variationSettings)
          .map(([token, value]) => `'${token}' ${value}`)
          .join(', ')
      }
    : {};

  return (
    <span
      className={cn(IconVariants({ variant, size, style, className }))}
      style={fontVariationStyle}
      {...props}
    >
      {children}
    </span>
  )
}

export { Icon, IconVariants }
