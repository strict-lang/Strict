using System;
using System.Collections.Generic;
using System.Text;

#pragma warning disable IDE0130 // Namespace does not match folder structure
namespace NCrunch.Framework;

public abstract class ResourceUsageAttribute(params string[] resourceName) : Attribute
{
	public string[] ResourceNames { get; } = resourceName;
}

public class ExclusivelyUsesAttribute(params string[] resourceName) : ResourceUsageAttribute(resourceName);